import torch
import torchmetrics
from sklearn.metrics import precision_recall_fscore_support


def reconstruct_image(sub_masks, center_coords, image_indices, image_sizes):
    # Implementation of the reconstruction logic
    # Combine sub-masks to reconstruct the original images separately
    # Handle overlap by taking the mean of overlapping pixels
    
    reconstructed_images = [torch.zeros(image_size) for image_size in image_sizes]
    count_map = [torch.zeros(image_size) for image_size in image_sizes]

    for sub_mask, center_coord, image_index in zip(sub_masks, center_coords, image_indices):
        x_center, y_center = center_coord

        # Calculate the start and end coordinates of the sub-mask
        x_start = x_center - sub_mask.shape[2] // 2
        x_end = x_start + sub_mask.shape[2]
        y_start = y_center - sub_mask.shape[1] // 2
        y_end = y_start + sub_mask.shape[1]

        # Handle overlap by taking the mean of overlapping pixels
        reconstructed_images[image_index][:, y_start:y_end, x_start:x_end] += sub_mask
        count_map[image_index][:, y_start:y_end, x_start:x_end] += 1

    # Divide by the count map to obtain the mean value
    for i in range(len(image_sizes)):
        reconstructed_images[i] /= count_map[i]
        
        # Threshold the reconstructed image to obtain binary masks
        reconstructed_images[i] = (reconstructed_images[i] > 0.5).float()

    return reconstructed_images


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
        reconstructed_images = reconstruct_image(predictions, coords, indexes, self.image_sizes)
        reconstructed_mask = reconstruct_image(targets, coords, indexes, self.image_sizes)
        
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