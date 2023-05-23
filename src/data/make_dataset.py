import os
import sys

parent = os.path.abspath(os.path.curdir)
sys.path.insert(1, parent)

import glob
import cv2
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

import numpy as np
from tiler import Tiler
from tqdm import tqdm
from PIL import Image

import json

from src.utils import get_device, get_mask_shape
from constant import (TRAIN_SAVE_PATH, TEST_SAVE_PATH, TRAIN_READ_PATH, TEST_READ_PATH,
                      TRAIN_FRAGMENTS_PATH, TEST_FRAGMENTS_PATH,
                      Z_START, Z_DIM, TILE_SIZE,
                      TRAIN_FRAGMENTS)

DEVICE = get_device()


class VesuviusDataset(Dataset):
    def __init__(self, fragments, test, augmentation, on_ram, save, read):
        self.fragments = fragments
        self.test = test
        self.augmentation = augmentation
        self.on_ram = on_ram
        self.save = save
        self.read = read

        self.set_path = TRAIN_FRAGMENTS_PATH if not test else TEST_FRAGMENTS_PATH
        self.save_path = TRAIN_SAVE_PATH if not test else TEST_SAVE_PATH
        self.read_path = TRAIN_READ_PATH if not test else TEST_READ_PATH
        self.tile_size = f'{TILE_SIZE}x{TILE_SIZE}'
        self.fragment_list = []
        self.bbox_list = {}
        self.indexes = []

        if self.save or not self.read:
            if on_ram == 'before':
                self.images = torch.ByteTensor().to(DEVICE)
                self.masks = torch.ByteTensor().to(DEVICE)
                self.bboxes = torch.IntTensor()

            for fragment in self.fragments:
                fragment_list, images, masks, bboxes, bbox_list = tile_fragment(self, fragment)
                self.bbox_list = {**self.bbox_list, **bbox_list}

                if on_ram == 'before':
                    self.fragment_list += fragment_list
                    self.images = torch.cat((self.images, images), dim=0)
                    self.masks = torch.cat((self.masks, masks), dim=0)
                    self.bboxes = torch.cat((self.bboxes, bboxes), dim=0)

            if self.save:
                save_bboxes(self.save_path, self.tile_size, self.bbox_list)

        elif self.read:
            for fragment in self.fragments:
                save_folder = os.path.join(self.read_path, self.tile_size, str(fragment))
                self.indexes += [os.path.join(str(fragment), d)
                                 for d in os.listdir(save_folder)
                                 if os.path.isdir(os.path.join(save_folder, d))]

                self.indexes = sorted(self.indexes)

        self.bbox_list = read_bboxes(self.read_path, self.tile_size)

        self.transforms = T.RandomApply(nn.ModuleList([T.RandomRotation(180),
                                                       T.RandomPerspective(),
                                                       T.ElasticTransform(alpha=500.0, sigma=10.0),
                                                       T.RandomHorizontalFlip(),
                                                       T.RandomVerticalFlip()]), p=0.5)

    def __len__(self):
        if self.read:
            length = len(self.indexes)
        else:
            length = len(self.images)

        return length

    def __getitem__(self, idx):
        if self.read:
            fragment = self.indexes[idx].split('/')[0]
            image_path = os.path.join(self.read_path, self.tile_size, self.indexes[idx], '*.png')
            image = torch.ByteTensor(build_3d_image(image_path, shape=(Z_DIM, TILE_SIZE, TILE_SIZE)))
            mask_path = os.path.join(self.read_path, self.tile_size, f'{self.indexes[idx]}.png')
            mask = torch.ByteTensor(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE))
            bbox = torch.IntTensor(self.bbox_list[self.indexes[idx]])
        else:
            fragment = self.fragment_list[idx]
            image = self.images[idx]
            mask = self.masks[idx]
            bbox = self.bboxes[idx]  # [x0, y0, x1, y1]

        image = image / 255.0
        mask = torch.unsqueeze(mask / 255.0, dim=0)

        if self.augmentation:
            seed = random.randint(0, 2 ** 32)
            torch.manual_seed(seed)
            image = self.transforms(image)
            torch.manual_seed(seed)
            mask = torch.squeeze(self.transforms(mask))

        return fragment, image, mask, bbox


def tile_fragment(self, fragment):
    set_path = self.set_path
    on_ram = self.on_ram
    save = self.save
    save_path = self.save_path
    tile_size = self.tile_size
    fragment_path = os.path.join(set_path, fragment)

    print(f'\nLoad slice images from fragment {fragment}...')
    image_path = os.path.join(fragment_path, 'surface_volume/*.tif')
    image_shape = get_mask_shape()
    image = build_3d_image(image_path, shape=(Z_DIM, image_shape[0], image_shape[1]))

    print(f'\nBuild image tiler from fragment {fragment}...')
    image_tiler, image_pad = create_image_tiler(image)

    print(f'\nLoad mask from fragment {fragment}...')
    mask_path = os.path.join(fragment_path, 'inklabels.png')
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    print(f'\nBuild mask tiler from fragment {fragment}...')
    mask_tiler, mask_pad = create_mask_tiler(mask)

    fragment_list = []
    bbox_list = {}
    images, masks, bboxes = init_tensors()
    tiles_zip = zip(image_tiler(image_pad), mask_tiler(mask_pad))

    print(f'\nExtract {TILE_SIZE}x{TILE_SIZE} tiles from fragment {fragment}...')
    for image_tile, mask_tile in tqdm(tiles_zip, total=image_tiler.n_tiles):
        if mask_tile[1].max() > 0:
            fragment_list.append(fragment)
            image, mask = get_image_mask_tile(image_tile, mask_tile)
            bbox = get_bbox_tile(image_tiler, image_tile)

            if save:
                save_tile(tile_size, fragment, image_tile, mask_tile, save_path)
                bbox_list[f'{fragment}/{image_tile[0]}'] = bbox.tolist()

            if on_ram == 'before':
                images = torch.cat((images, image), dim=0)
                masks = torch.cat((masks, mask), dim=0)
                bboxes = torch.cat((bboxes, torch.unsqueeze(bbox, dim=0)), dim=0)

    return fragment_list, images, masks, bboxes, bbox_list


def read_bboxes(path, tile_size):
    bboxes_path = os.path.join(path, tile_size, 'bboxes.json')
    with open(bboxes_path) as f:
        bbox_list = json.load(f)

    return bbox_list


def save_bboxes(save_path, tile_size, bbox_list):
    bboxes_path = os.path.join(save_path, tile_size, 'bboxes.json')
    with open(bboxes_path, 'w') as f:
        json.dump(bbox_list, f)


def build_3d_image(path, shape):
    image = np.zeros(shape=shape, dtype=np.uint8)
    image_path = sorted(glob.glob(path))[Z_START:Z_START + Z_DIM]
    for i, slice_path in enumerate(image_path):
        image[i, ...] = cv2.imread(slice_path, cv2.IMREAD_GRAYSCALE)

    return image


def create_image_tiler(image):
    image_tiler = Tiler(data_shape=image.shape,
                        tile_shape=(Z_DIM, TILE_SIZE, TILE_SIZE),
                        overlap=0.5,
                        channel_dimension=0)

    new_shape, padding = image_tiler.calculate_padding()
    image_tiler.recalculate(data_shape=new_shape)
    image_pad = np.pad(image, padding)

    return image_tiler, image_pad


def create_mask_tiler(mask):
    mask_tiler = Tiler(data_shape=mask.shape,
                       tile_shape=(TILE_SIZE, TILE_SIZE),
                       overlap=0.5)

    new_shape, padding = mask_tiler.calculate_padding()
    mask_tiler.recalculate(data_shape=new_shape)
    mask_pad = np.pad(mask, padding)

    return mask_tiler, mask_pad


def init_tensors():
    images = torch.ByteTensor().to(DEVICE)
    masks = torch.ByteTensor().to(DEVICE)
    bboxes = torch.IntTensor()

    return images, masks, bboxes


def get_image_mask_tile(image_tile, mask_tile):
    image = torch.unsqueeze(torch.from_numpy(image_tile[1]), dim=0).to(DEVICE)
    mask = torch.unsqueeze(torch.from_numpy(mask_tile[1]), dim=0).to(DEVICE)

    return image, mask


def get_bbox_tile(image_tiler, image_tile):
    bbox = image_tiler.get_tile_bbox(image_tile[0])
    bbox = torch.IntTensor([bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]])

    return bbox


def save_tile(tile_size, fragment, image_tile, mask_tile, save_path):
    fragment_str = str(fragment)
    tile_id = str(image_tile[0])
    save_folder = os.path.join(save_path, tile_size, fragment_str)
    os.makedirs(os.path.join(save_folder, tile_id), exist_ok=True)

    mask_pil = Image.fromarray(mask_tile[1])
    mask_pil.save(os.path.join(save_folder, f'{tile_id}.png'))

    for i in range(image_tile[1].shape[0]):
        image_pil = Image.fromarray(image_tile[1][i])
        image_pil.save(os.path.join(save_folder, tile_id, f'{i}.png'))


if __name__ == '__main__':
    train_dataset = VesuviusDataset(TRAIN_FRAGMENTS,
                                    test=False,
                                    augmentation=True,
                                    on_ram='after',
                                    save=False,
                                    read=True)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=16)

    for fragment, image, mask, bbox in train_dataloader:
        print(fragment)
        print(image.shape)
        print(mask.shape)
        print(bbox.shape)
        break
