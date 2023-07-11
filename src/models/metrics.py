import os
import sys

sys.path.insert(1, os.path.abspath(os.path.curdir))

import torch
import torchmetrics
from torchmetrics.classification import BinaryFBetaScore

import torchvision

import numpy as np

from src.constant import TRAIN_FRAGMENTS_PATH

import cv2


class F05Score(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.target = []
        self.preds = []
        self.add_state('target', default=[], dist_reduce_fx=None)
        self.add_state('preds', default=[], dist_reduce_fx=None)

    def update(self, target, preds):
        self.target.append(target)
        self.preds.append(preds)

    def compute(self):
        preds = torch.cat(self.preds, dim=0)
        target = torch.cat(self.target, dim=0)

        preds = preds.view(-1)
        target = target.view(-1)
        
        target = torch.where(target.view(-1) > 0.5, 1, 0)

        best_sub_f05_threshold = 0
        best_sub_f05_score = 0

        for threshold in np.arange(0.01, 1, 0.01):
            f05score = BinaryFBetaScore(0.5, threshold).to(device=self.device)

            sub_f05_score = f05score(preds, target)
            if best_sub_f05_score < sub_f05_score:
                best_sub_f05_threshold = np.float32(threshold)
                best_sub_f05_score = sub_f05_score.to(torch.float32)

        return best_sub_f05_threshold, best_sub_f05_score

    def reset(self):
        self.fragments = []
        self.bboxes = []
        self.target = []
        self.preds = []
