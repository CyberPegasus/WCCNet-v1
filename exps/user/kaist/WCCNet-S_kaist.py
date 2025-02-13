#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from wccnet.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.0
        self.width = 1.0
        self.num_classes = 1
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.max_epoch = 80
        self.warmup_epochs = 3
        self.no_aug_epochs = 20
        self.eval_interval = 2
        self.data_num_workers = 8
        self.input_size = (640, 640)  # (height, width)
        
        # -----Training Optimizing-----#
        self.min_lr_ratio = 0.2 
        self.basic_lr_per_img = 0.01 / 64.0 
        self.attn_lr = 0.0001
        self.weight_decay = 2e-2
        
        self.data_dir = 'datasets/kaist/'
        self.output_dir = "./exps_results"
        self.train_ann = "instances_train.json"
        self.val_ann = "instances_val.json"
        self.test_ann = "instances_test.json"
        
        # ----data transform--#
        self.mixup_prob = 1.0
        self.enable_mixup = True
        self.multiscale_range = 3 
        
        # -------testing------#
        self.test_size = (640, 640)
        self.test_conf = 0.01
        self.nmsthre = 0.65
        self.datasetName = 'KAIST'

    def get_model(self, sublinear=False):
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            from wccnet.models import WCCNet, WCCNet_backbone, WCCNetHead
            
            depth_scale=0.25
            backbone = WCCNet_backbone(depth=33, Backbone='Darknet_DualStream', Reverse_RGBIR=False, depth_scale=depth_scale,CSA=True,CE=True)
            head_channels = [128, 256, 512]
            head_channels = [int(i*depth_scale) for i in head_channels]
            head = WCCNetHead(self.num_classes, self.width, in_channels=head_channels, act="lrelu")
            self.model = WCCNet(backbone, head)
        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)

        return self.model

    def get_data_loader(
        self, batch_size, is_distributed, no_aug=False, cache_img=False, illu_label=False
    ):
        from wccnet.data import (
            COCODataset,
            TrainTransform_multispectral,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            MosaicDetection_Multispectral,
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
                preproc=TrainTransform_multispectral(
                    max_labels=20, 
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob),
                cache=cache_img,
                illu_label=illu_label,
            )

        dataset = MosaicDetection_Multispectral(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform_multispectral(
                max_labels=50, 
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
