import os
import sys
sys.path.insert(1, os.path.abspath(os.path.curdir))

import torch
import torchmetrics
from torchmetrics.classification import BinaryFBetaScore

from src.utils import reconstruct_images, get_device
from constant import TILE_SIZE, TRAIN_FRAGMENTS_PATH

import cv2


class F05Score(torchmetrics.Metric):
    def __init__(self, fragments_shape, threshold):
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
        self.f05score = BinaryFBetaScore(0.5, threshold)

    def update(self, fragments, bboxes, target, preds):
        self.fragments += fragments
        self.bboxes.append(bboxes)
        self.target.append(target)
        self.preds.append(preds)

    def compute(self):
        preds = torch.cat(self.preds, dim=0)
        target = torch.cat(self.target, dim=0)
        bboxes = torch.cat(self.bboxes, dim=0)

        # Reconstruct the original images from sub-target
        padding = TILE_SIZE // 4
        reconstructed_preds = reconstruct_images(preds, bboxes, self.fragments, self.fragments_shape, padding)
        # reconstructed_target = reconstruct_images(target, bboxes, self.fragments, self.fragments_shape, padding)

        device = get_device()
        vector_preds = torch.HalfTensor().to(device)
        vector_target = torch.HalfTensor().to(device)

        for fragment_id in self.fragments_shape.keys():
            mask_path = os.path.join(TRAIN_FRAGMENTS_PATH, fragment_id, 'mask.png')
            mask = torch.from_numpy(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE))
            print(mask.dtype, mask.shape, torch.min(mask), torch.max(mask), mask)
            reconstructed_pred = torch.where(mask == 255, reconstructed_preds[fragment_id], 0)
            vector_preds = torch.cat((vector_preds, reconstructed_pred.view(-1)), dim=0)

            target_path = os.path.join(TRAIN_FRAGMENTS_PATH, fragment_id, 'inklabels.png')
            target = torch.from_numpy(cv2.imread(target_path, cv2.IMREAD_GRAYSCALE) / 255.0).type(torch.HalfTensor)
            vector_target = torch.cat((vector_target, target.view(-1)), dim=0)

        # Calculate F0.5 score between sub images and sub label target
        preds = preds.view(-1)
        target = torch.where(target.view(-1) > 0.5, 1, 0)
        sub_f05_score = self.f05score(preds, target)

        # Calculate F0.5 score between reconstructed images and label target
        vector_target = torch.where(vector_target > 0.5, 1, 0)
        f05_score = self.f05score(vector_preds, vector_target)

        return f05_score, sub_f05_score

    def reset(self):
        self.fragments = []
        self.bboxes = []
        self.target = []
        self.preds = []
