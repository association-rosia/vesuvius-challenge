import os
import sys
sys.path.insert(1, os.path.abspath(os.path.curdir))

import torch
import torchmetrics
from torchmetrics.classification import BinaryFBetaScore

from src.utils import reconstruct_images


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
        self.f05score = BinaryFBetaScore(0.5, threshold).half()

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
        reconstructed_preds = reconstruct_images(preds, bboxes, self.fragments, self.fragments_shape)
        reconstructed_target = reconstruct_images(target, bboxes, self.fragments, self.fragments_shape)

        vector_preds = torch.Tensor().to(device=self.device)
        vector_target = torch.Tensor().to(device=self.device)

        for fragment_id in self.fragments_shape.keys():
            view_preds = reconstructed_preds[fragment_id].view(-1)
            vector_preds = torch.cat((view_preds, vector_preds), dim=0)
            view_target = reconstructed_target[fragment_id].view(-1)
            vector_target = torch.cat((view_target, vector_target), dim=0)

        preds = preds.view(-1)
        target = target.view(-1)

        print()
        print(torch.min(target), torch.max(target))
        print()

        # Calculate F0.5 score between sub images and sub label target
        sub_f05_score = self.f05score(preds, target)

        # Calculate F0.5 score between reconstructed images and label target
        f05_score = self.f05score(vector_preds, vector_target)

        return f05_score, sub_f05_score

    def to(self, device):
        super().to(device=device)
        self.f05score.to(device=device)
        return self

    def reset(self):
        self.fragments = []
        self.bboxes = []
        self.target = []
        self.preds = []



