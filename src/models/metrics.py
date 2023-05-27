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
        self.masks = []
        self.outputs = []
        self.add_state('fragments', default=[], dist_reduce_fx=None)
        self.add_state('bboxes', default=[], dist_reduce_fx=None)
        self.add_state('masks', default=[], dist_reduce_fx=None)
        self.add_state('outputs', default=[], dist_reduce_fx=None)

        self.fragments_shape = fragments_shape
        self.f05score = BinaryFBetaScore(0.5, threshold)

    def update(self, fragments, bboxes, masks, outputs):
        self.fragments += fragments
        self.bboxes.append(bboxes)
        self.masks.append(masks)
        self.outputs.append(outputs)

    def compute(self):
        outputs = torch.cat(self.outputs, dim=0)
        masks = torch.cat(self.masks, dim=0)
        bboxes = torch.cat(self.bboxes, dim=0)

        # Reconstruct the original images from sub-masks
        reconstructed_outputs = reconstruct_images(
            outputs, bboxes, self.fragments, self.fragments_shape
        )
        reconstructed_masks = reconstruct_images(
            masks, bboxes, self.fragments, self.fragments_shape
        )

        vector_outputs = torch.Tensor().to(device=self.device)
        vector_masks = torch.Tensor().to(device=self.device)

        for fragment_id in self.fragments_shape.keys():
            view_outputs = reconstructed_outputs[fragment_id].view(-1)
            vector_outputs = torch.cat(
                (view_outputs, vector_outputs), dim=0
            )

            view_masks = reconstructed_masks[fragment_id].view(-1)
            vector_masks = torch.cat((view_masks, vector_masks), dim=0)

        outputs = outputs.view(-1)
        masks = masks.view(-1)

        # Calculate F0.5 score between sub images and sub label masks
        print()
        print(outputs)
        print()
        print(masks)
        print()
        sub_f05_score = self.f05score(outputs, masks)
        print()
        print(sub_f05_score)
        print()

        # Calculate F0.5 score between reconstructed images and label masks
        f05_score = self.f05score(vector_outputs, vector_masks)

        return f05_score, sub_f05_score

    def to(self, device):
        super().to(device=device)
        self.f05score.to(device=device)
        return self

    def reset(self):
        self.fragments = []
        self.bboxes = []
        self.masks = []
        self.outputs = []



