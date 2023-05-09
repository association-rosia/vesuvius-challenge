import os, sys

parent = os.path.abspath(os.path.curdir)
sys.path.insert(1, parent)

import numpy as np

import torch
import pandas as pd

from torch.utils.data import DataLoader

from src.models.lightning import LightningVesuvius
from src.utils import reconstruct_images, get_device
from src.data.make_dataset import CustomDataset

from constant import MODELS_DIR, TEST_FRAGMENTS

MODELS_NAME = 'model_name.pt'
MODEL_PATH = os.path.join(MODELS_DIR, MODELS_NAME)

def main():

    # Set the device for inference
    device = get_device()

    # Load the trained model
    model = LightningVesuvius.load_from_checkpoint(MODEL_PATH)
    model = model.to(device)
    model.eval()
    
    test_dataloader = DataLoader(
        dataset=CustomDataset(TEST_FRAGMENTS),
        batch_size=1,
        )

    outputs = []
    center_coords = []
    image_sizes = []
    image_indices = []
    
    # Iterate over the test data
    for inputs, center_coord, image_size, image_indice in test_dataloader:
        center_coords.append(center_coord)
        image_sizes.append(image_size)
        image_indices.append(image_indice)
        
        # Move the batch to the device
        inputs = inputs.to(device)
        
        # Perform inference
        with torch.no_grad():
            outputs.append(model(inputs))
    
    # Reconstruct the original images from predicted masks
    reconstructed_images, image_indices = reconstruct_images(outputs, center_coords, image_sizes, image_indices, True)

    # Convert the reconstructed images to numpy array or tensor, depending on your needs
    reconstructed_images = reconstructed_images.cpu().numpy()
    
    submission_list = []
    
    for id, index in image_indices.items():
        rle_image = rle(reconstructed_images[index])
        submission_list.append([id, rle_image])
    
    # Create a DataFrame to store the results
    submission_df = pd.DataFrame(submission_list, columns=['Id', 'Predicted'])

    # Save the results to a CSV file
    submission_df.to_csv("submission.csv", index=False)
    
    
def rle(output):
    starts = np.array((output[:-1] == 0) & (output[1:] == 1))
    ends = np.array((output[:-1] == 1) & (output[1:] == 0))
    starts_ix = np.where(starts)[0] + 2
    ends_ix = np.where(ends)[0] + 2
    lengths = ends_ix - starts_ix
    return " ".join(map(str, sum(zip(starts_ix, lengths), ())))
