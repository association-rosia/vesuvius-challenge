import os, sys

parent = os.path.abspath(os.path.curdir)
sys.path.insert(1, parent)

import pandas as pd

import torch
import torchmetrics

from src.utils import reconstruct_images
from constant import TEST_FRAGMENTS


class Submission(torchmetrics.Metric):
    def __init__(self, image_sizes):
        super().__init__()
        self.add_state("predictions", default=[], dist_reduce_fx=None)
        self.add_state("coords", default=[], dist_reduce_fx=None)
        self.add_state("indexes", default=[], dist_reduce_fx=None)
        self.image_sizes = image_sizes

    def update(self, predictions, coords, indexes):
        self.predictions += predictions
        self.coords += coords
        self.indexes += indexes

    def compute(self):
        predictions = torch.cat(self.predictions, dim=0)
        coords = torch.cat(self.coords, dim=0)
        indexes = torch.cat(self.indexes, dim=0)
        
        # Reconstruct the original images from predicted masks
        reconstructed_images, image_indices = reconstruct_images(predictions, coords, indexes, self.image_sizes, True)
        
        submission_list = []
        
        for id, index in image_indices.items():
            rle_image = rle(reconstructed_images[index])
            submission_list.append([id, rle_image])
    
        # Create a DataFrame to store the results
        submission_df = pd.DataFrame(submission_list, columns=['Id', 'Predicted'])
        
        # Save the results to a CSV file
        submission_df.to_csv("submission.csv", index=False)
        
        return None

    def reset(self):
        self.predictions = []
        self.targets = []
        self.coords = []
        self.indexes = []
