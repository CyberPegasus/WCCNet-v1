#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# WCCNet is extended from YOLOX
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="6"
import sys
sys.path.append(os.getcwd())

import argparse,random,warnings,json
from loguru import logger

import torch
import torch.backends.cudnn as cudnn

from wccnet.core import Trainer, launch
from wccnet.exp import get_exp
from wccnet.utils import configure_nccl, configure_omp, get_num_devices


def make_parser():
    parser = argparse.ArgumentParser("WCCNet train parser")
    parser.add_argument("-cfg", "--config", type=str, default=None)
    return parser

# @logger.catch

def main(exp, args):
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # set environment variables for distributed training
    # configure_nccl()
    # configure_omp()
    
    cudnn.benchmark = True

    trainer = Trainer(exp, args)
    trainer.train()

if __name__ == "__main__":
    cfg_file = 'cfg/kaist/WCCNet_kaist.json'
    # True for args input, False for dict load
    init_mode = False
    if init_mode:
        args = make_parser().parse_args()
        with open(cfg_file,'w') as f:
            json.dump(args.__dict__,f)
    else:
        parser = argparse.ArgumentParser("WCCNet train parser")
        parser.add_argument("-cfg", "--config", type=str, default=None)
        args = parser.parse_args()
        if args.config is not None:
            cfg_file = args.config
        else:
            assert cfg_file is not None, "Please specify the path of the used cfg file."
        with open(cfg_file,'r') as f:
            args.__dict__=json.load(f)
    
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()

    dist_url = "auto" if args.dist_url is None else args.dist_url

    main(exp,args)
