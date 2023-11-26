import argparse, os, random
import torch
import numpy as np
import os
os.environ['TORCH_USE_CUDA_DSA'] = '1'

def parse_args():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--model', type=str, default="Model_LAV_RETNET", choices=["Model_LA", "Model_LAV","Model_LAV_Retnet_self_write"])
    parser.add_argument('--layer', type=int, default=3)
    parser.add_argument('--inner_layer', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--dropout_r', type=float, default=0.1)
    parser.add_argument('--multi_head', type=int, default=4)
    parser.add_argument('--ff_size', type=int, default=1024)
    parser.add_argument('--word_embed_size', type=int, default=768)
    parser.add_argument('--new_word_embed_size', type=int, default=512)
    parser.add_argument('--embed_dropout', type=float, default=0.0001)

    # Data
    parser.add_argument('--lang_seq_len', type=int, default=60)
    parser.add_argument('--audio_seq_len', type=int, default=60)
    parser.add_argument('--video_seq_len', type=int, default=60)
    parser.add_argument('--audio_feat_size', type=int, default=64)
    parser.add_argument('--video_feat_size', type=int, default=64)

    # Training
    parser.add_argument('--output', type=str, default='ckpt/')
    parser.add_argument('--name', type=str, default='exp0/')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epoch', type=int, default=20)
    parser.add_argument('--opt', type=str, default="Adam")
    parser.add_argument('--opt_params', type=str, default="{'betas': '(0.9, 0.98)', 'eps': '1e-9'}")
    parser.add_argument('--lr_base', type=float, default=0.0001)
    parser.add_argument('--lr_decay', type=float, default=0.5)
    parser.add_argument('--lr_decay_times', type=int, default=3)
    parser.add_argument('--warmup_epoch', type=float, default=0)
    parser.add_argument('--grad_norm_clip', type=float, default=-1)
    parser.add_argument('--eval_start', type=int, default=0)
    parser.add_argument('--early_stop', type=int, default=3)
    parser.add_argument('--seed', type=int, default=random.randint(0, 9999999))
    parser.add_argument('--multiway', type=bool, default=False, help="Specify the multiway parameter")
    parser.add_argument('--chunkwise_recurrent', type=bool, default=True, help="Specify the chunkwise_recurrent parameter")
    parser.add_argument('--recurrent_chunk_size', type=int, default=20, help="Specify the recurrent_chunk_size parameter")

    # Dataset and task
    parser.add_argument('--dataset', type=str, choices=['MELD', 'MOSEI'], default='MOSEI')
    parser.add_argument('--task', type=str, choices=['sentiment', 'emotion'], default='emotion')
    parser.add_argument('--task_binary', type=bool, default=False)
    parser.add_argument('--n_train', type=bool, default=True)
    parser.add_argument('--n_valid', type=bool, default=True)
    parser.add_argument('--n_test', type=bool, default=True)
    args = parser.parse_args()
    return args