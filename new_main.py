import argparse, os, random
import torch
from torch.utils.data import DataLoader
from mosei_dataset import Mosei_Dataset
from meld_dataset import Meld_Dataset
#from model_LA import Model_LA
# from model_LAV import Model_LAV
# from model_LAV_with_Retnet_self_write import Model_LAV_Retnet_self_write
from new_model_12 import Model_LAV_RETNET
from new_train import train
import numpy as np
from utils.compute_args import compute_args
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
from new_config import parse_args


if __name__ == '__main__':
    # Base on args given, compute new args
    args = compute_args(parse_args())

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # DataLoader
    train_dset = eval(args.dataloader)('train', args)
    eval_dset = eval(args.dataloader)('valid', args)
    train_loader = DataLoader(train_dset, args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_dset, args.batch_size, num_workers=4, pin_memory=True)

    # Net
    net = eval(args.model)(args).cuda()
    print("Total number of parameters : " + str(sum([p.numel() for p in net.parameters()]) / 1e6) + "M")
    net = net.cuda()

    # Create Checkpoint dir
    if not os.path.exists(os.path.join(args.output, args.name)):
        os.makedirs(os.path.join(args.output, args.name))

    # Run training
    eval_accuracies = train(net, train_loader, eval_loader, args)
