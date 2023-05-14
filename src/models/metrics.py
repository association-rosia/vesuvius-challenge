import os, sys

parent = os.path.abspath(os.path.curdir)
sys.path.insert(1, parent)

import numpy as np

import torch
import torchmetrics
from torchmetrics.classification import BinaryFBetaScore

from src.utils import reconstruct_images


class F05Score(torchmetrics.Metric):
    def __init__(self, mask_sizes, threshold):
        super().__init__()
        self.add_state("predictions", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)
        self.add_state("coords", default=[], dist_reduce_fx=None)
        self.add_state("indexes", default=[], dist_reduce_fx=None)
        
        self.mask_sizes = mask_sizes
        self.f05score = BinaryFBetaScore(0.5, threshold)

    def update(self, predictions, targets, coords, indexes):
        self.predictions.append(predictions)
        self.targets.append(targets)
        self.coords.append(coords)
        self.indexes += indexes

    def compute(self):
        predictions = torch.cat(self.predictions, dim=0)
        targets = torch.cat(self.targets, dim=0)
        coords = torch.cat(self.coords, dim=0)
        
        print(predictions.shape, targets.shape, coords.shape, self.indexes)
        
        # Reconstruct the original images from sub-masks
        reconstructed_predictions = reconstruct_images(predictions, coords, self.indexes, self.mask_sizes)
        reconstructed_targets = reconstruct_images(targets, coords, self.indexes, self.mask_sizes)
        
        vector_predictions = torch.Tensor()
        vector_targets = torch.Tensor()
        
        for fragment_id in self.mask_sizes.keys():
            new_shape = np.prod(reconstructed_predictions[fragment_id].shape)
            
            view_predictions =  reconstructed_predictions[fragment_id].view(new_shape, 1)
            vector_predictions = torch.cat(view_predictions, vector_predictions, dim=0)
            
            view_targets =  reconstructed_targets[fragment_id].view(new_shape, 1)
            vector_targets = torch.cat(view_targets, vector_targets, dim=0)
        
        predictions = predictions.view()
        targets = targets.view(start_dim=1)
        
        print(predictions.shape, targets.shape)
        
        # Calculate F0.5 score between sub images and sub label masks
        sub_f05_score = self.f05score(predictions, targets)
        
        # Calculate F0.5 score between reconstructed images and label masks
        f05_score = self.f05score(vector_predictions, vector_targets)
        
        return f05_score, sub_f05_score

    def reset(self):
        self.predictions = []
        self.targets = []
        self.coords = []
        self.indexes = []