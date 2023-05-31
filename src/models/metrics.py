import os
import sys

sys.path.insert(1, os.path.abspath(os.path.curdir))

import torch
import torchmetrics
from torchmetrics.classification import BinaryFBetaScore

import numpy as np

from src.utils import reconstruct_images, get_device
from constant import TILE_SIZE, TRAIN_FRAGMENTS_PATH

import cv2


class F05Score(torchmetrics.Metric):
    def __init__(self, fragments_shape):
        super().__init__()
        self.fragments = []
        self.bboxes = []
        self.target = []
        self.preds = []
        self.add_state('fragments', default=[], dist_reduce_fx=None)
        self.add_state('bboxes', default=[], dist_reduce_fx=None)
        self.add_state('target', default=[], dist_reduce_fx=None)
        self.add_state('preds', default=[], dist_reduce_fx=None)

        self.fragments_shape = fragments_shape

    def update(self, fragments, bboxes, target, preds):
        self.fragments += fragments
        self.bboxes.append(bboxes)
        self.target.append(target)
        self.preds.append(preds)

    def compute(self):
        preds = torch.cat(self.preds, dim=0)
        target = torch.cat(self.target, dim=0)
        bboxes = torch.cat(self.bboxes, dim=0)

        padding = TILE_SIZE // 4
        reconstructed_preds = reconstruct_images(preds, bboxes, self.fragments, self.fragments_shape, padding)

        device = get_device()
        vector_preds = torch.FloatTensor().to(device)
        vector_target = torch.FloatTensor().to(device)

        for fragment_id in self.fragments_shape.keys():
            mask_path = os.path.join(TRAIN_FRAGMENTS_PATH, fragment_id, 'mask.png')
            mask = torch.from_numpy(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)).to(device)
            reconstructed_pred = torch.where(mask == 255, reconstructed_preds[fragment_id], 0)
            vector_preds = torch.cat((vector_preds, reconstructed_pred.view(-1)), dim=0)

            target_path = os.path.join(TRAIN_FRAGMENTS_PATH, fragment_id, 'inklabels.png')
            loaded_target = torch.from_numpy(cv2.imread(target_path, cv2.IMREAD_GRAYSCALE) / 255.0)
            loaded_target = loaded_target.to(device)
            vector_target = torch.cat((vector_target, loaded_target.view(-1)), dim=0)

        # Calculate F0.5 score between sub images and sub label target
        preds = preds.view(-1)
        target = torch.where(target.view(-1) > 0.5, 1, 0)

        vector_target = torch.where(vector_target > 0.5, 1, 0)

        best_sub_f05_threshold = -1
        best_sub_f05_score = -1

        best_f05_threshold = -1
        best_f05_score = -1

        for threshold in np.arange(0.1, 1, 0.1):
            f05score = BinaryFBetaScore(0.5, threshold).to(device=device)

            sub_f05_score = f05score(preds, target)
            if best_sub_f05_threshold < sub_f05_score:
                best_sub_f05_threshold = threshold
                best_sub_f05_score = sub_f05_score

            f05_score = f05score(vector_preds, vector_target)
            if best_f05_threshold < f05_score:
                best_f05_threshold = threshold
                best_f05_score = f05_score

        return best_sub_f05_threshold, best_sub_f05_score, best_f05_threshold, best_f05_score

    def reset(self):
        self.fragments = []
        self.bboxes = []
        self.target = []
        self.preds = []
