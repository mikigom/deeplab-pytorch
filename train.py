#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-11-01

from __future__ import absolute_import, division, print_function

import os.path as osp

import click
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from addict import Dict
from tensorboardX import SummaryWriter
from torchnet.meter import MovingAverageValueMeter
from tqdm import tqdm
import numpy as np

from libs.colors import color_mapping_on_batch
from mpl.mpl import MaxPoolingLoss
from libs.datasets import get_dataset
from libs.models import DeepLabV3Plus_ResNet101_MSC, DeepLabV2_ResNet101_MSC
from libs.utils.loss import CrossEntropyLoss2d


def get_params(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            yield param

"""
def get_params(model, key):
    # For Dilated FCN
    if key == "1x":
        for m in model.named_modules():
            if "layer" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    for p in m[1].parameters():
                        yield p
    # For conv weight in the ASPP module
    if key == "10x":
        for m in model.named_modules():
            if "aspp" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    yield m[1].weight
    # For conv bias in the ASPP module
    if key == "20x":
        for m in model.named_modules():
            if "aspp" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    yield m[1].bias
"""


def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter, max_iter, power):
    if iter % lr_decay_iter or iter > max_iter:
        return None
    new_lr = init_lr * (1 - float(iter) / max_iter) ** power
    optimizer.param_groups[0]["lr"] = new_lr


@click.command()
@click.option("-c", "--config", type=str, required=True)
@click.option("--cuda/--no-cuda", default=True)
def main(config, cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    if cuda:
        current_device = torch.cuda.current_device()
        print("Running on", torch.cuda.get_device_name(current_device))
    else:
        print("Running on CPU")

    # Configuration
    CONFIG = Dict(yaml.load(open(config)))

    dataset = get_dataset(CONFIG.DATASET)(
        data_path=CONFIG.ROOT,
        crop_size=256,
        scale=(0.6, 0.8, 1., 1.2, 1.4),
        rotation=15,
        flip=True,
        mean=(CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R),
    )
    """
    # Dataset 10k or 164k
    dataset = get_dataset(CONFIG.DATASET)(
        root=CONFIG.ROOT,
        split=CONFIG.SPLIT.TRAIN,
        base_size=513,
        crop_size=CONFIG.IMAGE.SIZE.TRAIN,
        mean=(CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R),
        warp=CONFIG.WARP_IMAGE,
        scale=(0.5, 0.75, 1.0, 1.25, 1.5),
        flip=True,
    )
    """

    # DataLoader
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=CONFIG.BATCH_SIZE.TRAIN,
        num_workers=CONFIG.NUM_WORKERS,
        shuffle=True,
    )
    loader_iter = iter(loader)

    # Model
    model = DeepLabV3Plus_ResNet101_MSC(n_classes=CONFIG.N_CLASSES)
    state_dict = torch.load(CONFIG.INIT_MODEL)
    model.load_state_dict(state_dict, strict=False)  # Skip "aspp" layer
    model = nn.DataParallel(model)
    model.to(device)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    # Optimizer
    optimizer = torch.optim.Adam(
                params=get_params(model.module),
                lr=CONFIG.LR,
                weight_decay=CONFIG.WEIGHT_DECAY,
    )
    """
    # Optimizer
    optimizer = torch.optim.SGD(
        # cf lr_mult and decay_mult in train.prototxt
        params=[
            {
                "params": get_params(model.module, key="1x"),
                "lr": CONFIG.LR,
                "weight_decay": CONFIG.WEIGHT_DECAY,
            },
            {
                "params": get_params(model.module, key="10x"),
                "lr": 10 * CONFIG.LR,
                "weight_decay": CONFIG.WEIGHT_DECAY,
            },
            {
                "params": get_params(model.module, key="20x"),
                "lr": 20 * CONFIG.LR,
                "weight_decay": 0.0,
            },
        ],
        momentum=CONFIG.MOMENTUM,
    )
    """
    # Loss definition
    criterion = CrossEntropyLoss2d(ignore_index=CONFIG.IGNORE_LABEL)
    criterion.to(device)
    max_pooling_loss = MaxPoolingLoss(ratio=0.3, p=1.7, reduce=True)

    # TensorBoard Logger
    writer = SummaryWriter(CONFIG.LOG_DIR)
    loss_meter = MovingAverageValueMeter(20)

    model.train()
    model.module.scale.freeze_bn()

    for iteration in tqdm(
        range(1, CONFIG.ITER_MAX + 1),
        total=CONFIG.ITER_MAX,
        leave=False,
        dynamic_ncols=True,
    ):

        """
        # Set a learning rate
        poly_lr_scheduler(
            optimizer=optimizer,
            init_lr=CONFIG.LR,
            iter=iteration - 1,
            lr_decay_iter=CONFIG.LR_DECAY,
            max_iter=CONFIG.ITER_MAX,
            power=CONFIG.POLY_POWER,
        )
        """

        # Clear gradients (ready to accumulate)
        optimizer.zero_grad()

        iter_loss = 0
        for i in range(1, CONFIG.ITER_SIZE + 1):
            try:
                images, labels = next(loader_iter)
            except:
                loader_iter = iter(loader)
                images, labels = next(loader_iter)

            images = images.to(device)
            labels = labels.to(device).unsqueeze(1).float()

            # Propagate forward
            logits = model(images)

            # Loss
            loss = 0
            for logit in logits:
                # Resize labels for {100%, 75%, 50%, Max} logits
                labels_ = F.interpolate(labels, logit.shape[2:], mode="nearest")
                labels_ = labels_.squeeze(1).long()
                # Compute NLL and MPL
                nll_loss = criterion(logit, labels_)
                # loss += nll_loss
                loss += max_pooling_loss(nll_loss)

            # Backpropagate (just compute gradients wrt the loss)
            loss /= float(CONFIG.ITER_SIZE)
            loss.backward()

            iter_loss += float(loss)

        loss_meter.add(iter_loss)

        # Update weights with accumulated gradients
        optimizer.step()

        if iteration % CONFIG.ITER_TB == 0:
            writer.add_scalar("train_loss", loss_meter.value()[0], iteration)
            for i, o in enumerate(optimizer.param_groups):
                writer.add_scalar("train_lr_group{}".format(i), o["lr"], iteration)

            gt_viz, images_viz, predicts_viz = make_vizs(images, labels_, logits,
                                                        (CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R))
            writer.add_image("gt/images", torch.from_numpy(images_viz[0]), iteration)
            writer.add_image("gt/labels", torch.from_numpy(gt_viz[0]), iteration)
            for i, predict_viz in enumerate(predicts_viz):
                writer.add_image("predict/"+str(i), torch.from_numpy(predict_viz[0]), iteration)

            if False:  # This produces a large log file
                for name, param in model.named_parameters():
                    name = name.replace(".", "/")
                    writer.add_histogram(name, param, iteration, bins="auto")
                    if param.requires_grad:
                        writer.add_histogram(
                            name + "/grad", param.grad, iteration, bins="auto"
                        )

        # Save a model
        if iteration % CONFIG.ITER_SAVE == 0:
            torch.save(
                model.module.state_dict(),
                osp.join(CONFIG.SAVE_DIR, "checkpoint_{}.pth".format(iteration)),
            )

        # Save a model (short term)
        if iteration % 100 == 0:
            torch.save(
                model.module.state_dict(),
                osp.join(CONFIG.SAVE_DIR, "checkpoint_current.pth"),
            )

    torch.save(
        model.module.state_dict(), osp.join(CONFIG.SAVE_DIR, "checkpoint_final.pth")
    )


def make_vizs(images, labels_, logits, mean):
    viz_number = 1
    # TensorBoard
    images_viz = images.data.cpu().numpy()[:viz_number] + np.array(mean)[np.newaxis, :, np.newaxis, np.newaxis]
    predicts_viz = []
    for logit in logits:
        predicts_viz.append(torch.max(logit, 1)[1].data.cpu().numpy()[:viz_number])
    gt_viz = labels_.data.cpu().numpy()[:viz_number]
    for i, predict_viz in enumerate(predicts_viz):
        predicts_viz[i] = color_mapping_on_batch(predict_viz)
    gt_viz = color_mapping_on_batch(gt_viz)
    return gt_viz, images_viz/255., predicts_viz


if __name__ == "__main__":
    main()
