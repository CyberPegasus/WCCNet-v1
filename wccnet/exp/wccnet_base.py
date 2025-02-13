#!/usr/bin/env python3
# -*- coding:utf-8 -*-
 

import os
import random

import torch
import torch.distributed as dist
import torch.nn as nn

from .base_exp import BaseExp
from pycocotools.coco import COCO

class Exp(BaseExp):
    def __init__(self):
        super().__init__()

        # ---------------- model config ---------------- #
        self.num_classes = 80
        self.depth = 1.00
        self.width = 1.00
        self.act = 'silu'

        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        self.data_num_workers = 10
        self.input_size = (640, 640)  # (height, width)
        # Actual multiscale ranges: [640-5*32, 640+5*32].
        # To disable multiscale training, set the
        # self.multiscale_range to 0.
        self.multiscale_range = 3
        # You can uncomment this line to specify a multiscale range
        # self.random_size = (14, 26)
        self.data_dir = None
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        self.test_ann = "instances_test2017.json"

        # --------------- transform config ----------------- #
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.degrees = 10.0
        self.translate = 0.1
        self.mosaic_scale = (0.1, 2)
        self.mixup_scale = (0.5, 1.5)
        self.shear = 2.0
        self.enable_mixup = True

        # --------------  training config --------------------- #
        self.warmup_epochs = 5
        self.max_epoch = 300
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.01 / 64.0 #Note: default to 0.01/64.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 15
        self.min_lr_ratio = 0.05
        self.ema = True

        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 10
        self.eval_interval = 10
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.gradient_clipping = False # Custom TODO: realize gradient clipping
        self.attn_lr = 0.0001
         
        # -----------------  testing config ------------------ #
        self.test_size = (640, 640)
        self.test_conf = 0.01
        self.nmsthre = 0.65
        self.datasetName = 'KAIST'

    def get_model(self):
        from wccnet.models import WCCNet, WCCNet_backbone, WCCNetHead

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = WCCNet_backbone(self.depth, self.width, in_channels=in_channels, act=self.act)
            head = WCCNetHead(self.num_classes, self.width, in_channels=in_channels, act=self.act)
            self.model = WCCNetHead(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def get_data_loader(
        self, batch_size, is_distributed, no_aug=False, cache_img=False
    ):
        from wccnet.data import (
            COCODataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            MosaicDetection,
            worker_init_reset_seed,
        )
        from wccnet.utils import (
            wait_for_the_master,
            get_local_rank,
        )

        local_rank = get_local_rank()

        with wait_for_the_master(local_rank):
            dataset = COCODataset(
                data_dir=self.data_dir,
                json_file=self.train_ann,
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob),
                cache=cache_img,
            )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def random_resize(self, data_loader, epoch, rank, is_distributed):
        tensor = torch.LongTensor(2).cuda()

        if rank == 0:
            size_factor = self.input_size[1] * 1.0 / self.input_size[0]
            if not hasattr(self, 'random_size'):
                min_size = int(self.input_size[0] / 64) - self.multiscale_range
                max_size = int(self.input_size[0] / 64) + self.multiscale_range
                self.random_size = (min_size, max_size)
            size = random.randint(*self.random_size)
            size = (int(64 * size), 64 * int(size * size_factor))
            tensor[0] = size[0]
            tensor[1] = size[1]

        if is_distributed:
            dist.barrier()
            dist.broadcast(tensor, 0)

        input_size = (tensor[0].item(), tensor[1].item())
        return input_size

    def preprocess(self, rgb, ir, targets, tsize):
        scale_y = tsize[0] / self.input_size[0]
        scale_x = tsize[1] / self.input_size[1]
        if scale_x != 1 or scale_y != 1:
            rgb = nn.functional.interpolate(
                rgb, size=tsize, mode="bilinear", align_corners=False
            )
            ir = nn.functional.interpolate(
                ir, size=tsize, mode="bilinear", align_corners=False
            )
            targets[..., 1::2] = targets[..., 1::2] * scale_x
            targets[..., 2::2] = targets[..., 2::2] * scale_y
        return rgb, ir, targets
    
    # 返回所有Module模块，即在init中出现的、继承了module类的模块
    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2, pg3 = [], [], [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases, no decay
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)  # BatchNorm no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    # if 'bifusion' in k:
                    #     print(k)
                    pg1.append(v.weight)  # fc and conv apply decay
                elif hasattr(v,"in_proj_bias") and isinstance(v.in_proj_bias, nn.Parameter):
                    pg3.append(v.in_proj_bias) 
                    if hasattr(v,"in_proj_weight") and isinstance(v.in_proj_weight, nn.Parameter):
                        pg3.append(v.in_proj_weight)
                    elif hasattr(v,"in_proj_weight_q") and isinstance(v.in_proj_weight_q, nn.Parameter):
                        pg3.append(v.in_proj_weight_q)
                    elif hasattr(v,"in_proj_weight_k") and isinstance(v.in_proj_weight_k, nn.Parameter):
                        pg3.append(v.in_proj_weight_k)
                    elif hasattr(v,"in_proj_weight_v") and isinstance(v.in_proj_weight_v, nn.Parameter):
                        pg3.append(v.in_proj_weight_v)

            
            optimizer = torch.optim.SGD(
                pg0, lr=lr, momentum=self.momentum, nesterov=True
            )
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )  # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2})
            optimizer.add_param_group({"params": pg3, "lr": self.attn_lr})
            self.optimizer = optimizer

        return self.optimizer

    def get_lr_scheduler(self, lr, iters_per_epoch):
        from wccnet.utils import LRScheduler

        scheduler = LRScheduler(
            self.scheduler,
            lr,
            iters_per_epoch,
            self.max_epoch,
            warmup_epochs=self.warmup_epochs,
            warmup_lr_start=self.warmup_lr,
            no_aug_epochs=self.no_aug_epochs,
            min_lr_ratio=self.min_lr_ratio,
        )
        return scheduler

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from wccnet.data import COCODataset, ValTransform

        valdataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann if not testdev else self.test_ann,
            name="val" if not testdev else "test",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
            "shuffle":False,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset,**dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from wccnet.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
            per_class_AP = True,
            datasetName = self.datasetName,
        )
        return evaluator

    def eval(self, model, evaluator, is_distributed, half=False):
        return evaluator.evaluate(model, is_distributed, half)
