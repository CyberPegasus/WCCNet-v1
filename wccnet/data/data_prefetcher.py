#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# WCCNet is extended from YOLOX

import torch


class DataPrefetcher:
    """
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader, illu_aware:bool=False):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.input_cuda = self._input_cuda_for_image
        self.record_stream = DataPrefetcher._record_stream_for_image
        self.illu_aware = illu_aware
        self.preload()

    def preload(self):
        if self.illu_aware:
            try:
                self.next_img_rgb, self.next_img_ir, self.next_target, _, _, self.next_illu = next(self.loader)
            except StopIteration:
                self.next_img_rgb = None
                self.next_img_ir = None
                self.next_target = None
                self.next_illu = None
                return
            with torch.cuda.stream(self.stream):
                self.input_cuda()
                self.next_target = self.next_target.cuda(non_blocking=True)
                self.next_illu = self.next_illu.cuda(non_blocking=True)   
        
        else:
            try:
                self.next_img_rgb, self.next_img_ir, self.next_target, _, _ = next(self.loader)
            except StopIteration:
                self.next_img_rgb = None
                self.next_img_ir = None
                self.next_target = None
                return    
            with torch.cuda.stream(self.stream):
                self.input_cuda()
                self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input_rgb = self.next_img_rgb
        input_ir = self.next_img_ir
        target = self.next_target
        if self.illu_aware:
            illu = self.next_illu
            
        if input_rgb is not None:
            self.record_stream(input_rgb)
        if input_ir is not None:
            self.record_stream(input_ir)
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        if self.illu_aware and illu is not None:
            illu.record_stream(torch.cuda.current_stream())
        self.preload()
        if not self.illu_aware:
            return input_rgb, input_ir, target
        else:
            return input_rgb, input_ir, target, illu

    def _input_cuda_for_image(self):
        self.next_img_rgb = self.next_img_rgb.cuda(non_blocking=True)
        self.next_img_ir = self.next_img_ir.cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())
