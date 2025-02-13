#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# WCCNet is extended from YOLOX

import argparse
import os
import random
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from wccnet.core import launch
from wccnet.exp import get_exp
from wccnet.utils import configure_nccl, fuse_model, get_local_rank, get_model_info, setup_logger
import json

def make_parser():
    parser = argparse.ArgumentParser("WCCNet Eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
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
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--test",
        dest="test",
        default=False,
        action="store_true",
        help="Evaluating on test-dev set.",
    )
    parser.add_argument(
        "--speed",
        dest="speed",
        default=False,
        action="store_true",
        help="speed test only.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


@logger.catch
def main(exp, args, num_gpu):
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed testing. This will turn on the CUDNN deterministic setting, "
        )

    cudnn.benchmark = True

    rank = get_local_rank()
    is_distributed = False
    file_name = os.path.join(exp.output_dir, args.experiment_name)

    if rank == 0:
        os.makedirs(file_name, exist_ok=True)

    setup_logger(file_name, distributed_rank=rank, filename="val_log.txt", mode="a")
    logger.info("Args: {}".format(args))
    gpu_info = torch.cuda.get_device_properties(args.gpuid)
    logger.info(gpu_info)
    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    logger.info("Model Structure:\n{}".format(str(model)))

    evaluator = exp.get_evaluator(args.batch_size, is_distributed, True, args.legacy)
    # set is_distributed to False
    evaluator.per_class_AP = True
    evaluator.per_class_AR = True

    torch.cuda.set_device(args.gpuid)
    model.cuda(args.gpuid)
    model.eval()

    if not args.speed and not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint from {}".format(ckpt_file))
        loc = "cuda:{}".format(args.gpuid)
        ckpt = torch.load(ckpt_file, map_location=loc)
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    trt_file = None
    decoder = None

    # start evaluate
    # set is_distributed to False
    *_, summary = evaluator.evaluate(
        model, is_distributed, args.fp16, trt_file, decoder, exp.test_size, args.result_path+'/'+args.experiment_name+'.json'
    )
    logger.info("\n" + summary)


if __name__ == "__main__":
    # input configuration file's relative path
    # e.g. "cfg/kaist/WCCNet.json"
    cfg_file = 'cfg/flir/WCCNet_flir_eval.json'
    
    # init_mode default to False
    init_mode = False
    if init_mode:
        args = make_parser().parse_args()
        with open(cfg_file,'w') as f:
            json.dump(args.__dict__,f)
    else:
        parser = argparse.ArgumentParser("WCCNet eval parser")
        parser.add_argument("-cfg", "--config", type=str, default=None)
        parser.add_argument(
            "--easycfg",
            dest="easycfg",
            default=True,
            action="store_false",
            help="Using training cfg for eval",
        )
        
        args = parser.parse_args()
        if args.config is not None:
            cfg_file = args.config
        else:
            assert cfg_file is not None, 'If not using ArgumentParser, should set the string value for cfg_file'
            
        easycfg = args.easycfg
        with open(cfg_file,'r') as f:
            args.__dict__=json.load(f)
            if easycfg:
                assert args.multigpu == False and args.devices == 1
                eval_dict = {"conf":None,"nms":None,"tsize":None, "test":False,"legacy":False,"trt":False,"speed":False,"fuse":False,"result_path": "outputs/predictions/"}
                args.__dict__.update(eval_dict)
            
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    dist_url = "auto" if args.dist_url is None else args.dist_url
    
    # evaluation code is only implemented with single GPU
    # The GPU of gpuid specific in configuration file
    main(exp, args, num_gpu=1)
