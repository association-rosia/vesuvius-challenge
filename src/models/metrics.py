import os
import sys

sys.path.insert(1, os.path.abspath(os.path.curdir))

import torch
import torchmetrics
from torchmetrics.classification import BinaryFBetaScore

import numpy as np

from src.utils import reconstruct_outputs
from src.constant import TRAIN_FRAGMENTS_PATH

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
        reconstructed_preds = reconstruct_outputs(preds, bboxes, self.fragments, self.fragments_shape)

        vector_preds = torch.Tensor().to(device=self.device)
        vector_target = torch.Tensor().to(device=self.device)

        for fragment_id in self.fragments_shape.keys():
            mask_path = os.path.join(TRAIN_FRAGMENTS_PATH, fragment_id, 'mask.png')
            mask = torch.from_numpy(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)).to(device=self.device)
            reconstructed_pred = torch.where(mask == 255, reconstructed_preds[fragment_id], 0)
            vector_preds = torch.cat((vector_preds, reconstructed_pred.view(-1)), dim=0)

            target_path = os.path.join(TRAIN_FRAGMENTS_PATH, fragment_id, 'inklabels.png')
            loaded_target = torch.from_numpy(cv2.imread(target_path, cv2.IMREAD_GRAYSCALE) / 255.0)
            loaded_target = loaded_target.to(torch.float32).to(self.device)
            vector_target = torch.cat((vector_target, loaded_target.view(-1)), dim=0)

        # Calculate F0.5 score between sub images and sub label target
        preds = preds.view(-1)
        target = torch.where(target.view(-1) > 0.5, 1, 0)
        vector_target = torch.where(vector_target > 0.5, 1, 0)

        best_sub_f05_threshold = 0
        best_f05_threshold = 0
        best_sub_f05_score = 0
        best_f05_score = 0

        for threshold in np.arange(0.01, 1, 0.01):
            f05score = BinaryFBetaScore(0.5, threshold).to(device=self.device)

            sub_f05_score = f05score(preds, target)
            if best_sub_f05_score < sub_f05_score:
                best_sub_f05_threshold = np.float32(threshold)
                best_sub_f05_score = sub_f05_score.to(torch.float32)

            f05_score = f05score(vector_preds, vector_target)
            if best_f05_score < f05_score:
                best_f05_threshold = np.float32(threshold)
                best_f05_score = f05_score.to(torch.float32)

        return best_sub_f05_threshold, best_sub_f05_score, best_f05_threshold, best_f05_score

    def reset(self):
        self.fragments = []
        self.bboxes = []
        self.target = []
        self.preds = []
