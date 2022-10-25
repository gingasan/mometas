from __future__ import absolute_import, division, print_function
import argparse
import json
import logging
import os
import pickle
import gzip
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.functional import gumbel_softmax
from tqdm import tqdm, trange
from transformers import AutoTokenizer
from transformers import SchedulerType, get_scheduler
from modeling_bert import BertForMultiObjectiveLM
from processor import Processor


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


PTOS = ["mlm", "atd", "dtd", "cse"]


class Mometas(object):
    def __init__(self, model, processor, log_dir, lr=1e-1, gamma=2, k=100):
        self.model = model
        self.processor = processor
        self.log_dir = log_dir

        self.device = model.device
        self.pstate = torch.ones(len(PTOS), device=model.device)
        self.pstate.requires_grad_()
        self.lstate = [0.0 for _ in range(len(PTOS))]
        self.rollout = []

        self.lr = lr
        self.gamma = gamma
        self.k = k
        self.meta_step = 0

    def __call__(self):
        sample = gumbel_softmax(self.pstate.clone().detach(), hard=True).argmax().item()
        self.rollout.append(sample)
        return sample

    def init_state(self, inputs):
        val_losses = self.test_all(inputs)
        self.lstate[:] = val_losses[:]

    def update(self, inputs):
        val_losses = self.test_all(inputs)
        reward = sum([individual_rewarding(ba, lo) for ba, lo in zip(self.lstate, val_losses)])
        self.lstate[:] = val_losses[:]

        p = torch.softmax(self.pstate, dim=-1)
        tmp_losses = []
        for sample in self.rollout:
            a = torch.zeros([len(PTOS)], dtype=torch.int, device=self.device)
            a[sample] = 1
            tmp_losses += [(-reward * p.log() * a).sum()]
        meta_loss = sum(tmp_losses) - self.gamma * -(p * p.log()).sum()
        meta_loss.backward()

        self.pstate = (self.pstate - self.lr * self.pstate.grad.clone().detach()).detach()

        self.meta_step += 1
        self.pstate.requires_grad_()
        del self.rollout[:]

        return meta_loss, reward

    def test_all(self, inputs):
        self.model.eval()
        val_losses = []
        for ob, batch in zip(PTOS, list(inputs)):
            model_inputs = {}
            model_inputs["input_ids"] = batch[0]
            model_inputs["attention_mask"] = batch[1]
            model_inputs = self.processor(ob, **model_inputs)
            model_inputs = {_: t.to(self.device) for _, t in model_inputs.items() if t is not None}

            with torch.no_grad():
                outputs = self.model(**model_inputs, flow=ob)
                loss = outputs["loss"]
            val_losses += [loss.mean().data]

        return val_losses

    def logging(self):
        log_file = os.path.join(self.log_dir, "weights.txt")
        p = torch.softmax(self.pstate.clone().detach(), dim=-1).cpu().numpy()
        with open(log_file, "a") as f:
            f.write("\t".join([str(round(w, 3)) for w in p]) + "\n")


class C4DataMaster(object):
    def __init__(self, data_dir):
        self.input_files = []
        for root, _, files in os.walk(data_dir):
            for filename in files:
                self.input_files.append(os.path.join(root, filename))
        self.input_files.sort()

        self.p = 0
        self.n = len(self.input_files)

    def get_num_splits(self):
        return self.n

    def convert_corpus_to_features(self, max_seq_length, tokenizer, input_file=None):
        if input_file is None:
            input_file = self.input_files[self.p]
            self.p = (self.p + 1) % self.n

        directory, filename = os.path.split(input_file)
        cached_features_file =  os.path.split(tokenizer.name_or_path)[-1] + "_" + str(max_seq_length) + "_" + filename.split('.')[0] + ".cache"
        cached_features_file = os.path.join(directory, "../.cache/", cached_features_file)
        os.makedirs(os.path.join(directory, "../.cache/"), exist_ok=True)

        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s...", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                features = pickle.load(handle)
        else:
            features = []
            with gzip.open(input_file, "rb") as f:
                for i, line in tqdm(enumerate(f)):
                    line = json.loads(line.strip())
                    encoded_inputs = tokenizer(line["text"],
                                               padding="max_length",
                                               max_length=max_seq_length,
                                               truncation=True)

                    input_ids = encoded_inputs["input_ids"]
                    input_mask = encoded_inputs["attention_mask"]
                    tokens = tokenizer.convert_ids_to_tokens(input_ids)
                    features.append((input_ids, input_mask))

                    if i < 5:
                        logger.info("*** Example ***")
                        logger.info("guid: %s" % i)
                        logger.info("tokens: %s" % " ".join(tokens))
                        logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                        logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))

            logger.info("Saving features into cached file %s...", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)

        all_input_ids = torch.tensor([f[0] for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f[1] for f in features], dtype=torch.long)

        return TensorDataset(all_input_ids, all_input_mask)


def individual_rewarding(a, b):
    return max((a - b + 1e-6) / (a + 1e-6) * (abs(a - b) > 1e-3), -1)


def k_shot_sampling(data, bsz, k=1):
    indices = np.random.choice(range(len(data) - 1), bsz * k)
    sampler = SubsetRandomSampler(indices)
    return DataLoader(data, sampler=sampler, batch_size=bsz)


def main():
    parser = argparse.ArgumentParser()

    # Data config.
    parser.add_argument("--train_data_dir", type=str, default="../corpus/c4",
                        help="Input directiory of training corpus with numbers of sub-files.")
    parser.add_argument("--eval_data_file", type=str, default="../corpus/c4/00000.jsonl.gz",
                        help="Validation file for meta-test.")
    parser.add_argument("--load_model_path", type=str, default="bert-large-uncased",
                        help="Path to load the pre-trained language model.")
    parser.add_argument("--cache_dir", type=str, default="../cache/",
                        help="Directory to store the pre-trained language models downloaded from s3.")
    parser.add_argument("--output_dir", type=str, default="model/",
                        help="Output directory to save training results.")

    # Training config.
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to train the model.")
    parser.add_argument("--do_prepro", action="store_true",
                        help="Whether to do preprocessing first on training corpus instead of training the model.")
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--block_size", type=int, default=256,
                        help="Maximum total input sequence length after word-piece tokenization.")
    parser.add_argument("--per_gpu_train_batch_size", type=int, default=64,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", type=int, default=128,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Peak learning rate for Adam.")
    parser.add_argument("--noise_probability", type=float, default=0.15,
                        help="Mask probability for masked language modeling.")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                        help="Total number of epochs to go.")
    parser.add_argument("--max_train_steps", type=int, default=50000,
                        help="Total number of training steps to go.")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear",
                        help="Scheduler type for learning rate warmup.")
    parser.add_argument("--warmup_proportion", type=float, default=0.06,
                        help="Proportion of training to perform learning rate warmup for.")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="L2 weight decay for training.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Threshold to apply gradient clipping.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Whether not to use CUDA when available.")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use mixed precision.")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for initialization.")
    parser.add_argument("--save_steps", type=int, default=10000,
                        help="How many steps to save the checkpoint once.")
    
    args = parser.parse_args()
    
    args.local_rank = int(os.environ["LOCAL_RANK"])
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend="nccl")
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, args.local_rank != -1, args.fp16))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    torch.save(args, os.path.join(args.output_dir, "train_args.bin"))

    cache_dir = args.cache_dir
    tokenizer = AutoTokenizer.from_pretrained(args.load_model_path,
                                              do_lower_case=args.do_lower_case,
                                              cache_dir=cache_dir,
                                              use_fast=True)
    
    if args.do_prepro:
        dm = C4DataMaster(args.train_data_dir)
        for input_file in dm.input_files:
            dm.convert_corpus_to_features(args.block_size, tokenizer, input_file)
        if args.local_rank == 0:
            torch.distributed.barrier()

    if args.do_train:
        model = BertForMultiObjectiveLM.from_pretrained(args.load_model_path,
                                                        return_dict=True,
                                                        cache_dir=cache_dir)
        model.to(device)
        if args.local_rank == 0:
            torch.distributed.barrier()

        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
        if args.local_rank != -1:
            model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        model_uw = model.module if hasattr(model, "module") else model
        model_uw.resize_token_embeddings(len(tokenizer))

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            }
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_scheduler(name=args.lr_scheduler_type,
                                  optimizer=optimizer,
                                  num_warmup_steps=args.max_train_steps * args.warmup_proportion,
                                  num_training_steps=args.max_train_steps)

        if args.fp16:
            from torch.cuda.amp import autocast, GradScaler

            scaler = GradScaler()

        def save_checkpoint():
            output_dir = os.path.join("checkpoint-{}".format(global_step))
            os.makedirs(output_dir, exist_ok=True)

            logger.info("Saving model checkpoint to %s...", output_dir)
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

        args.train_batch_size = args.per_gpu_train_batch_size * max(1, n_gpu)
        data_master = C4DataMaster(args.train_data_dir)

        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, n_gpu)
        eval_features = data_master.convert_corpus_to_features(args.block_size, tokenizer, args.eval_data_file)

        if args.local_rank == 0:
            torch.distributed.barrier()

        processor = Processor(tokenizer, args.block_size)
        mometas = Mometas(model, processor, args.output_dir)

        test_dataloader = k_shot_sampling(eval_features, args.eval_batch_size, len(PTOS))
        mometas.init_state(test_dataloader)

        train_dataloader = None
        global_step = 0
        for epoch in trange(data_master.get_num_splits() * args.num_train_epochs, desc="Epoch"):
            if train_dataloader is not None:
                del train_dataloader
            
            if args.local_rank not in [-1, 0]:
                torch.distributed.barrier()

            train_features = data_master.convert_corpus_to_features(args.block_size, tokenizer)
            train_sampler = RandomSampler(train_features) if args.local_rank == -1 else DistributedSampler(train_features)
            train_dataloader = DataLoader(train_features, sampler=train_sampler, batch_size=args.train_batch_size)

            if args.local_rank == 0:
                torch.distributed.barrier()

            logger.info("***** Running epoch {} *****".format(epoch))
            logger.info("  Num examples = %d", len(train_features))
            logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
            logger.info("  Total train batch size = %d", args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
            logger.info("  Gradient accumulation steps = %d", args.gradient_accumulation_steps)
            logger.info("  Total optimization steps = %d", int(len(train_dataloader) // args.gradient_accumulation_steps))
            logger.info("  Global step = %d", global_step)

            train_loss = 0.0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                if args.local_rank not in [-1, 0]:
                    torch.distributed.barrier()

                sample = mometas()
                
                if args.local_rank == 0:
                    torch.distributed.barrier()
                    
                ob = PTOS[sample]
                model.train()
                
                inputs = {}
                inputs["input_ids"] = batch[0]
                inputs["attention_mask"] = batch[1]
                inputs = processor(ob, **inputs)
                inputs = {_: t.to(device) for _, t in inputs.items() if t is not None}

                if args.fp16:
                    with autocast():
                        outputs = model(**inputs,
                                        flow=ob)
                else:
                    outputs = model(**inputs,
                                    flow=ob)

                loss = outputs["loss"]
                if n_gpu > 1:
                    loss = loss.mean()
                if args.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                train_loss += loss.item()

                if args.fp16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if global_step >= args.max_train_steps:
                    break

                if global_step % mometas.k == 0:
                    if args.local_rank in [-1, 0]:
                        mometas.logging()

                    mometas.update(test_dataloader)

                    del test_dataloader
                    test_dataloader = k_shot_sampling(eval_features, args.eval_batch_size, len(PTOS))

                if (args.local_rank == -1 or torch.distributed.get_rank() == 0) and global_step % args.save_steps == 0:
                    save_checkpoint()

        if (args.local_rank == -1 or torch.distributed.get_rank() == 0):
            save_checkpoint()


if __name__ == "__main__":
    main()
