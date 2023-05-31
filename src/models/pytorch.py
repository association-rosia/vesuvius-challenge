import os
import sys

sys.path.insert(1, os.path.abspath(os.path.curdir))

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

from torchmetrics import MeanMetric

from src.data.make_dataset import DatasetVesuvius
from src.models.losses import BCEDiceLoss
from src.models.metrics import F05Score
from src.models.unet3d import Unet3d
from constant import TRAIN_FRAGMENTS, VAL_FRAGMENTS, TILE_SIZE, Z_DIM
from src.utils import get_device, get_fragments_shape

from tqdm import tqdm

DEVICE = get_device()
BATCH_SIZE = 8

training_dataset = DatasetVesuvius(
    fragments=TRAIN_FRAGMENTS,
    tile_size=TILE_SIZE,
    num_slices=Z_DIM,
    random_slices=False,
    selection_thr=0.01,
    augmentation=True,
    device=DEVICE
)

training_loader = DataLoader(
    dataset=training_dataset,
    batch_size=BATCH_SIZE,
    # shuffle=True,
    drop_last=True,
)

val_dataset = DatasetVesuvius(
    fragments=VAL_FRAGMENTS,
    tile_size=TILE_SIZE,
    num_slices=Z_DIM,
    random_slices=False,
    selection_thr=0.01,
    augmentation=True,
    device=DEVICE
)

val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=BATCH_SIZE,
    drop_last=True,
)

print()

model = Unet3d(nb_blocks=2, inputs_size=TILE_SIZE).to(DEVICE).half()
optimizer = AdamW(model.parameters(), lr=0.001)
# loss_fn = BCEDiceLoss(bce_weight=1)
loss_fn = nn.BCEWithLogitsLoss()
metric = F05Score(get_fragments_shape(VAL_FRAGMENTS, TILE_SIZE)).to(DEVICE)

training_loss = MeanMetric().to(DEVICE)
for i, batch in enumerate(training_loader):
    print(i)
    _, _, masks, images = batch
    masks = masks.to(DEVICE).half()
    images = images.to(DEVICE).half()

    optimizer.zero_grad()
    outputs = model(images)
    loss = loss_fn(outputs, masks)
    loss.backward()
    optimizer.step()
    # training_loss.update(loss)

epoch_loss = training_loss.compute()
