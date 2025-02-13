#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# WCCNet is extended from YOLOX

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

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
    parser.add_argument("-expn", "--experiment-name", type=str, default='exps/user/yolofpn.py')
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "-g", "--gpuid", default=None, type=int, help="GPU ID for training"
    )
    parser.add_argument(
        "-f","--exp_file",
        default=None,
        type=str,
        help="plz input your experiment description file",
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="resume training"
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument(
        "-e",
        "--start_epoch",
        default=None,
        type=int,
        help="resume training start epoch",
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision training.",
    )
    # 得放float16来量化，不然更放不下参数了
    parser.add_argument(
        "--cache",
        dest="cache",
        default=False,
        action="store_true",
        help="Caching imgs to RAM for fast training.",
    )
    parser.add_argument(
        "-o",
        "--occupy",
        dest="occupy",
        default=False,
        action="store_true",
        help="occupy GPU memory first for training.",
    )
    # 通过计算剩余内存空间并提前分配tensor来占据，然后del，来获得完整连续的内存空间
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
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
    configure_nccl()
    configure_omp()
    cudnn.benchmark = True

    trainer = Trainer(exp, args)
    trainer.train()

if __name__ == "__main__":
    cfg_file= 'cfg/flir/WCCNet_flir_multiGPU.json'
    # True for args input, False for dict load
    init_mode = False
    if init_mode:
        args = make_parser().parse_args()
        with open(cfg_file,'w') as f:
            json.dump(args.__dict__,f)
    else:
        parser = argparse.ArgumentParser("WCCNet train parser")
        args= parser.parse_args()
        with open(cfg_file,'r') as f:
            args.__dict__=json.load(f)
    
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    print(get_num_devices())
    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()

    dist_url = "auto" if args.dist_url is None else args.dist_url

    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args),
    )
