import os, sys

parent = os.path.abspath(os.path.curdir)
sys.path.insert(1, parent)

import torch
import torchmetrics
from sklearn.metrics import precision_recall_fscore_support

from src.utils import reconstruct_images


class F05Score(torchmetrics.Metric):
    def __init__(self, image_sizes):
        super().__init__()
        self.add_state("predictions", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)
        self.add_state("coords", default=[], dist_reduce_fx=None)
        self.add_state("indexes", default=[], dist_reduce_fx=None)
        self.image_sizes = image_sizes

    def update(self, predictions, targets, coords, indexes):
        self.predictions += predictions
        self.targets += targets
        self.coords += coords
        self.indexes += indexes

    def compute(self):
        predictions = torch.cat(self.predictions, dim=0)
        targets = torch.cat(self.targets, dim=0)
        coords = torch.cat(self.coords, dim=0)
        indexes = torch.cat(self.indexes, dim=0)
        
        # Reconstruct the original images from sub-masks
        reconstructed_images = reconstruct_images(predictions, coords, indexes, self.image_sizes)
        reconstructed_mask = reconstruct_images(targets, coords, indexes, self.image_sizes)
        
        # Calculate F0.5 score between sub images and sub label masks
        _, _, sub_f05_score, _ = precision_recall_fscore_support(predictions, targets, beta=0.5)
        
        # Calculate F0.5 score between reconstructed images and label masks
        _, _, f05_score, _ = precision_recall_fscore_support(reconstructed_images, reconstructed_mask, beta=0.5)
        
        return f05_score, sub_f05_score

    def reset(self):
        self.predictions = []
        self.targets = []
        self.coords = []
        self.indexes = []