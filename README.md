# Multi-Objective Meta-Sampler

This repo is for the EMNLP 2022 paper [*Forging Multiple Training Objectives for Pre-trained Language Models via Meta-Learning*](https://arxiv.org/abs/2210.10293).

## Pre-training corpus and pre-trained model
* Our per-training corpus: [a subset of C4](https://drive.google.com/file/d/1uZHd9cITakWaHKJs3p1W_WbKVNUzhe9k/view?usp=share_link).
* Our pre-trained model: [BERT-large](https://drive.google.com/file/d/1ex274wgvLH14xxhkxBWAycRU1vS7ldXQ/view?usp=share_link).

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
@inproceedings{DBLP:conf/emnlp/WuDZCXHZ22,
  author       = {Hongqiu Wu and
                  Ruixue Ding and
                  Hai Zhao and
                  Boli Chen and
                  Pengjun Xie and
                  Fei Huang and
                  Min Zhang},
  editor       = {Yoav Goldberg and
                  Zornitsa Kozareva and
                  Yue Zhang},
  title        = {Forging Multiple Training Objectives for Pre-trained Language Models
                  via Meta-Learning},
  booktitle    = {Findings of the Association for Computational Linguistics: {EMNLP}
                  2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022},
  pages        = {6454--6466},
  publisher    = {Association for Computational Linguistics},
  year         = {2022},
  url          = {https://aclanthology.org/2022.findings-emnlp.482},
  timestamp    = {Tue, 07 Feb 2023 17:10:52 +0100},
  biburl       = {https://dblp.org/rec/conf/emnlp/WuDZCXHZ22.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

