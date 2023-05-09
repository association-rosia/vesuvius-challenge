import torch
import numpy as np


def reconstruct_images(sub_masks, center_coords, image_ids, image_sizes, return_indices: bool = False):
    # Implementation of the reconstruction logic
    # Combine sub-masks to reconstruct the original images separately
    # Handle overlap by taking the mean of overlapping pixels
    
    reconstructed_images = [torch.zeros(image_size) for image_size in image_sizes]
    count_map = [torch.zeros(image_size) for image_size in image_sizes]
    image_indices = {image_id: i for i, image_id in enumerate(np.unique(image_ids))}
    

    for sub_mask, center_coord, image_id in zip(sub_masks, center_coords, image_ids):
        x_center, y_center = center_coord

        # Calculate the start and end coordinates of the sub-mask
        x_start = x_center - sub_mask.shape[2] // 2
        x_end = x_start + sub_mask.shape[2]
        y_start = y_center - sub_mask.shape[1] // 2
        y_end = y_start + sub_mask.shape[1]

        # Handle overlap by taking the mean of overlapping pixels
        reconstructed_images[image_indices[image_id]][:, y_start:y_end, x_start:x_end] += sub_mask
        count_map[image_indices[image_id]][:, y_start:y_end, x_start:x_end] += 1

    # Divide by the count map to obtain the mean value
    for i in range(len(image_sizes)):
        reconstructed_images[i] /= count_map[i]
        
        # Threshold the reconstructed image to obtain binary masks
        reconstructed_images[i] = (reconstructed_images[i] > 0.5).float()
    
    return (reconstructed_images, image_indices) if return_indices else reconstructed_images


def get_device():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    return torch.device(device=device)