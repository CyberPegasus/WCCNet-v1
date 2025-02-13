#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# WCCNet is extended from YOLOX

import os
from loguru import logger
import copy
import cv2
import numpy as np
from pycocotools.coco import COCO

from ..dataloading import get_wccnet_datadir
from .datasets_wrapper import Dataset


def remove_useless_info(coco):
    """
    Remove useless info in coco dataset. COCO object is modified inplace.
    This function is mainly used for saving memory (save about 30% mem).
    """
    if isinstance(coco, COCO):
        dataset = coco.dataset
        dataset.pop("info", None)
        dataset.pop("licenses", None)
        for img in dataset["images"]:
            img.pop("license", None)
            img.pop("coco_url", None)
            img.pop("date_captured", None)
            img.pop("flickr_url", None)
        for anno in coco.dataset["annotations"]:
            anno.pop("segmentation", None)


class COCODataset(Dataset):
    """
    COCO dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        json_file="instances_train2017.json",
        name="train",
        img_size=(416, 416),
        preproc=None,
        cache=False,
        illu_label:bool=False,
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train' or 'val')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        super().__init__(img_size)
        if data_dir is None:
            data_dir = os.path.join(get_wccnet_datadir(), "COCO")
        self.data_dir = data_dir
        self.json_file = json_file
        self.illu_label = illu_label # whether return the illumination label

        self.coco = COCO(os.path.join(self.data_dir, "annotations", self.json_file))
        remove_useless_info(self.coco)
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in cats])
        self.imgs = None
        # self.imgs_ir = None
        self.name = name
        self.img_size = img_size
        self.preproc = preproc
        self.annotations = self._load_coco_annotations()
        if cache:
            self._cache_images()

    def __len__(self):
        return len(self.ids)

    def __del__(self):
        del self.imgs

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def _cache_images(self):
        logger.warning(
            "\n********************************************************************************\n"
            "You are using cached images in RAM to accelerate training.\n"
            "This requires large system RAM.\n"
            "Make sure you have 200G+ RAM and 136G available disk space for training COCO.\n"
            "********************************************************************************\n"
        )
        max_h = self.img_size[0]
        max_w = self.img_size[1]
        cache_file = self.data_dir + "/img_resized_cache_" + self.name + ".array"
        # cache_file_ir = self.data_dir + "/imgir_resized_cache_" + self.name + ".array"
        if not os.path.exists(cache_file):
            logger.info(
                "Caching images for the first time. This might take about 20 minutes for COCO"
            )
            self.imgs = np.memmap(
                cache_file,
                shape=(len(self.ids), 2, max_h, max_w, 3),
                dtype=np.uint8,
                mode="w+",
            )
            # self.imgs_ir = np.memmap(
            #     cache_file_ir,
            #     shape=(len(self.ids), max_h, max_w, 3),
            #     dtype=np.uint8,
            #     mode="w+",
            # )
            from tqdm import tqdm
            from multiprocessing.pool import ThreadPool

            NUM_THREADs = min(8, os.cpu_count())
            loaded_images = ThreadPool(NUM_THREADs).imap(
                lambda x: self.load_resized_img(x),
                range(len(self.annotations)),
            )
            pbar = tqdm(enumerate(loaded_images), total=len(self.annotations))
            for k, out in pbar:
                self.imgs[k][0][: out[0].shape[0], : out[0].shape[1], :] = out[0].copy()
                self.imgs[k][1][: out[1].shape[0], : out[1].shape[1], :] = out[1].copy()
            self.imgs.flush()
            pbar.close()
        else:
            logger.warning(
                "You are using cached imgs! Make sure your dataset is not changed!!\n"
                "Everytime the self.input_size is changed in your exp file, you need to delete\n"
                "the cached data and re-generate them.\n"
            )

        logger.info("Loading cached imgs...")
        self.imgs = np.memmap(
            cache_file,
            shape=(len(self.ids), 2, max_h, max_w, 3),
            dtype=np.uint8,
            mode="r+",
        )

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 5))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        assert "file_name" in im_ann and "file_name_ir" in im_ann
        file_name = (
            im_ann["file_name"]
        )
        
        file_name_ir = (
            im_ann["file_name_ir"]
        )

        return (res, img_info, resized_info, file_name, file_name_ir)

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_resized_img(self, index):
        img, img_ir = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        
        resized_img_ir = cv2.resize(
            img_ir,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        
        return resized_img, resized_img_ir

    def load_image(self, index):
        file_name, file_name_ir = self.annotations[index][3], self.annotations[index][4]

        img_file_rgb = os.path.join(self.data_dir, self.name, file_name)
        img_file_ir = os.path.join(self.data_dir, self.name, file_name_ir)

        img_rgb = cv2.imread(img_file_rgb)
        img_ir = cv2.imread(img_file_ir)
        assert img_rgb is not None, f'read failed for: {self.data_dir} {self.name} {file_name}'
        assert img_ir is not None, f'read failed for: {self.data_dir} {self.name} {file_name_ir}'

        return img_rgb, img_ir

    def pull_item(self, index):
        id_ = self.ids[index]

        res, img_info, resized_info, set_name, _ = self.annotations[index]
        if self.illu_label:
            flag = False
            for i in ['set00','set01','set02','set06','set07','set08']:
                if i in set_name:
                    flag=True
                    break
            #is_day = np.array([1.0,0.0]) if (set_name in ['set00','set01','set02']) else np.array([0.0,1.0])
            is_day = 1 if flag else 0

        if self.imgs is not None:
            # assert False, 'cached img not implemented for rgb/ir detection'
            pad_img = self.imgs[index][0]
            pad_img_ir = self.imgs[index][1]
            img = pad_img[: resized_info[0], : resized_info[1], :].copy()
            img_ir = pad_img_ir[: resized_info[0], : resized_info[1], :].copy()
        else:
            img, img_ir = self.load_resized_img(index)
        
        if self.illu_label:
            return img, img_ir, res.copy(), img_info, np.array([id_]), is_day
        else:
            return img, img_ir, res.copy(), img_info, np.array([id_])

    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w.
                h, w (int): original shape of the image
            img_id (int): same as the input index. Used for evaluation.
        """
        if self.illu_label:
            img, img_ir, target, img_info, img_id, is_day = self.pull_item(index)
        else:
            img, img_ir, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, img_ir, target = self.preproc(img, img_ir, target, self.input_dim)
        
        if not self.illu_label:
            return img, img_ir, target, img_info, img_id
        else:
            return img, img_ir, target, img_info, img_id, is_day
