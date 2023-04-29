import os, sys

parent = os.path.abspath(os.path.curdir)
sys.path.insert(1, parent)

import glob
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
from tiler import Tiler

from constant import FRAGMENTS_PATH, TRAIN_FRAGMENTS, Z_START, Z_DIM, TILE_SIZE, DEVICE


def tile_fragment(fragment):
    fragment_path = os.path.join(FRAGMENTS_PATH, fragment)
    slices_path = sorted(glob.glob(os.path.join(fragment_path, 'surface_volume/*.tif')))[Z_START:Z_START + Z_DIM]
    slices = [cv2.imread(slice_path, cv2.IMREAD_GRAYSCALE) / 255.0 for slice_path in slices_path]
    slices = np.stack(slices, axis=0)

    slices_tiler = Tiler(data_shape=slices.shape,
                         tile_shape=(Z_DIM, TILE_SIZE, TILE_SIZE),
                         overlap=0.5,
                         channel_dimension=0)

    new_shape, padding = slices_tiler.calculate_padding()
    slices_tiler.recalculate(data_shape=new_shape)
    slices_pad = np.pad(slices, padding)

    ink_path = os.path.join(FRAGMENTS_PATH, fragment, 'inklabels.png')
    ink = cv2.imread(ink_path, cv2.IMREAD_GRAYSCALE) / 255.0

    ink_tiler = Tiler(data_shape=ink.shape,
                      tile_shape=(TILE_SIZE, TILE_SIZE),
                      overlap=0.5)

    new_shape, padding = ink_tiler.calculate_padding()
    ink_tiler.recalculate(data_shape=new_shape)
    ink_pad = np.pad(ink, padding)

    slices_list = []
    ink_list = []
    tiles_zip = zip(slices_tiler(slices_pad), ink_tiler(ink_pad))

    for slices_tile, ink_tile in tiles_zip:
        if ink_tile[1].max() > 0:
            # for the multi-context dataset we have to create a bigger padded
            # image to retrieve bigger image from the center of the current tile
            tile_bbox = slices_tiler.get_tile_bbox(slices_tile[0])
            slices_list.append(torch.from_numpy(slices_tile[1].astype('float16')))
            ink_list.append(torch.from_numpy(ink_tile[1].astype('float16')))

    slices = torch.stack(slices_list, dim=0).to(DEVICE)
    ink = torch.stack(ink_list, dim=0).to(DEVICE)

    return slices, ink


class CustomDataset(Dataset):
    def __init__(self, fragments):
        self.slices = torch.HalfTensor().to(DEVICE)
        self.ink = torch.HalfTensor().to(DEVICE)

        for fragment in fragments:
            slices, ink = tile_fragment(fragment)
            self.slices = torch.cat((self.slices, slices), dim=0)
            self.ink = torch.cat((self.ink, ink), dim=0)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        slices = torch.unsqueeze(self.slices[idx], dim=0)
        ink = torch.unsqueeze(torch.unsqueeze(self.ink[idx], dim=0), dim=0)

        return slices, ink


if __name__ == '__main__':
    train_dataset = CustomDataset(TRAIN_FRAGMENTS)
    slices, ink = train_dataset[0]
    print(slices.shape, ink.shape)
    
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=8,
        shuffle=False,
        )
    for batch_slices, batch_ink in train_dataloader:
        print(batch_slices.shape, batch_ink.shape)
        break
