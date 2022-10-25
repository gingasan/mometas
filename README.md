# Multi-Objective Meta-Sampler

This repo is for the EMNLP 2022 paper [*Forging Multiple Training Objectives for Pre-trained Language Models via Meta-Learning*](https://arxiv.org/abs/2210.10293).

## Pre-training corpus and pre-trained model
* Our per-training corpus: [a subset of C4]().
* Our pre-trained model: [BERT-large]().

## Run pre-training
Pytorch distributed training:
```bash
python -m torch.distributed.run --nproc_per_node 8 run_pretraining.py \
  --do_train \
  --do_prepro \
  --train_data_dir c4 \
  --eval_data_file c4/00099.jsonl.gz \
  --load_model_path bert-large-uncased \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 128 \
  --do_lower_case \
  --fp16 \
  --max_train_steps 50000
```
For the first commit, we recommend to first finish pre-processing all the training data on CPU:
```bash
python run_pretraining.py \
  --do_prepro \
  --train_data_dir c4 \
  --load_model_path bert-base-uncased \
  --do_lower_case
```

**Citation**
```bib
@article{DBLP:journals/corr/abs-2210-10293,
  author    = {Hongqiu Wu and
               Ruixue Ding and
               Hai Zhao and
               Boli Chen and
               Pengjun Xie and
               Fei Huang and
               Min Zhang},
  title     = {Forging Multiple Training Objectives for Pre-trained Language Models
               via Meta-Learning},
  journal   = {CoRR},
  volume    = {abs/2210.10293},
  year      = {2022},
  url       = {https://doi.org/10.48550/arXiv.2210.10293},
  doi       = {10.48550/arXiv.2210.10293},
  eprinttype = {arXiv},
  eprint    = {2210.10293},
  timestamp = {Mon, 24 Oct 2022 18:10:06 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2210-10293.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

