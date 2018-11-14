#!/usr/bin/env python
# coding: utf-8
#
# Author:   Junghoon Seo
# Created:  2018-11-12

from __future__ import print_function

import os.path as osp
import cv2
import math
import random
import numpy as np
import glob
from torch.utils import data


class DSTL(data.Dataset):
    """COCO-Stuff base class"""

    def __init__(
        self,
        data_path,
        crop_size=256,
        scale=(0.8, 0.9, 1.0, 1.1, 1.2),
        rotation=15,
        flip=True,
    ):
        self.data_path = data_path
        self.crop_size = crop_size
        self.scale = tuple(sorted(scale))
        self.rotation = rotation
        self.flip = flip

        self.files = []
        self.images = []
        self.labels = []
        self.ignore_label = None

        self._set_files()

        assert 90 % self.rotation == 0

    def _transform(self, image, label):
        scale_factor = 1
        if self.scale is not None:
            assert image.shape[0] * self.scale[0] * math.sqrt(2) / 2 > self.crop_size
            scale_factor = random.choice(self.scale)

        large_border_scale = 1
        small_border_scale = 1
        if self.rotation is not None:
            thetas = [k * self.rotation for k in range(int(90/self.rotation))]
            theta = math.radians(random.choice(thetas))
            large_border_scale = math.cos(theta) + math.sin(theta)
            small_border_scale = math.fabs(math.sin(theta) - math.cos(theta))

        if self.flip:
            # Random flipping
            if random.random() < 0.5:
                image = np.fliplr(image).copy()  # HWC
                label = np.fliplr(label).copy()  # HW

        if self.rotation is not None:
            k = random.choice([0, 1, 2, 3])
            image = np.rot90(image, k=k)
            label = np.rot90(label, k=k)

        base_h, base_w = label.shape
        large_patch_border_from_center = int(large_border_scale * scale_factor * self.crop_size / 2)
        small_patch_border_from_center = int(small_border_scale * scale_factor * self.crop_size / 2)

        center_h = random.randint(large_patch_border_from_center, base_h - large_patch_border_from_center)
        center_w = random.randint(large_patch_border_from_center, base_w - large_patch_border_from_center)

        src = np.float32([[center_w + small_patch_border_from_center, center_h - large_patch_border_from_center],
                          [center_w + large_patch_border_from_center, center_h + small_patch_border_from_center],
                          [center_w - large_patch_border_from_center, center_h - small_patch_border_from_center]]
                         )

        dst = np.float32([[self.crop_size, 0],
                          [self.crop_size, self.crop_size],
                          [0, 0]]
                         )

        M = cv2.getAffineTransform(src, dst)
        image = cv2.warpAffine(image, M, (self.crop_size, self.crop_size))
        label = cv2.warpAffine(label, M, (self.crop_size, self.crop_size), flags=cv2.INTER_NEAREST)

        # HWC -> CHW
        image = image.transpose(2, 0, 1)/255.
        return image, label

    def _set_files(self):
        # Set paths
        image_path = list(glob.iglob(osp.join(self.data_path, '*.npy'), recursive=False))
        label_path = list(glob.iglob(osp.join(self.data_path, '*_label.npy'), recursive=False))
        image_path = list(set(image_path) - set(label_path))
        image_path.sort()
        label_path.sort()
        assert len(image_path) == len(label_path)

        self.files = [(image_path[i], label_path[i]) for i in range(len(image_path))]

    def _load_data(self, index):
        image_path, label_path = self.files[index]
        # image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        image = np.load(image_path)
        label = np.load(label_path).astype(np.int64)
        return image, label

    def __getitem__(self, index):
        image, label = self._load_data(index)
        image, label = self._transform(image, label)
        return image.astype(np.float32), label.astype(np.int64)

    def __len__(self):
        return len(self.files)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        fmt_str += "    Location: {}\n".format(self.data_path)
        return fmt_str


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from torchvision.utils import make_grid
    from tqdm import tqdm

    kwargs = {"nrow": 5, "padding": 50}
    batch_size = 25

    dataset_root = "/mnt/nas/Dataset/dstl/numpy"
    dataset = DSTL(data_path=dataset_root)

    print(dataset)

    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for i, (images, labels) in tqdm(enumerate(loader), total=np.ceil(len(dataset) / batch_size), leave=False):
        if i == 0:
            image = make_grid(images, pad_value=-1, **kwargs).numpy()
            image = np.transpose(image, (1, 2, 0))
            mask = np.zeros(image.shape[:2])
            mask[(image != -1)[..., 0]] = 1.
            image = np.dstack((image, mask))

            labels = labels[:, np.newaxis, ...]
            label = make_grid(labels, pad_value=1, **kwargs).numpy()
            label_ = np.transpose(label, (1, 2, 0))[..., 0].astype(np.float32)
            label = cm.jet_r(label_ / 11.)
            mask = np.zeros(label.shape[:2])
            label[..., 3][(label_ == 1)] = 0
            label = label.astype(np.uint8)

            tiled_images = np.hstack((image, label))
            # cv2.imwrite("./docs/data.png", tiled_images)
            plt.imshow(np.dstack((tiled_images[..., 2::-1], tiled_images[..., 3])))
            plt.show()
            break
