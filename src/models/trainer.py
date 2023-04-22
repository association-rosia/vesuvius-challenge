from os.path import join
from tqdm import tqdm
from typing import Tuple
from datetime import datetime

from sklearn.metrics import precision_recall_fscore_support

import torch
from torch import nn
from torch.utils.data import DataLoader

import wandb


class Trainer:
    """ Define the Trainer class.

    :param model: our deep learning model
    :type model: nn.Module
    :param train_dataloader: training dataloader
    :type train_dataloader: DataLoader
    :param val_dataloader: validation dataloader
    :type val_dataloader: DataLoader
    :param epochs: max number of epochs
    :type epochs: int
    :param criterion: loss function
    :param optimizer: model optimizer
    :param scheduler: learning scheduler
    """
    def __init__(self, model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader,
                 epochs: int, criterion, optimizer, scheduler):
        self.model = model
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        self.criterion = criterion
        self.epochs = epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.timestamp = int(datetime.now().timestamp())
        self.val_best_r2_score = 0.

    def train_one_epoch(self) -> float:
        """ Train the model for one epoch.

        :return: the training loss
        :rtype: float
        """
        train_loss = 0.

        self.model.train()

        pbar = tqdm(self.train_loader, leave=False)
        for i, data in enumerate(pbar):
            keys_input = ['s_input', 'm_input', 'g_input']
            inputs = {key: data[key] for key in keys_input}
            labels = data['target']

            # Zero gradients for every batch
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.model(inputs)

            # Compute the loss and its gradients
            loss = self.criterion(outputs, labels)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            train_loss += loss.item()
            epoch_loss = train_loss / (i + 1)

            # Update the progress bar with new metrics values
            pbar.set_description(f'TRAIN - Batch: {i + 1}/{len(self.train_loader)} - '
                                 f'Epoch Loss: {epoch_loss:.5f} - '
                                 f'Batch Loss: {loss.item():.5f}')

        train_loss /= len(self.train_loader)

        return train_loss

    def val_one_epoch(self) -> Tuple[float, float, float, float]:
        """ Validate the model for one epoch.

        :return: the validation loss, the R^2 score and the aggregated R^2 score
        :rtype: tuple[float, float, float]
        """
        val_loss = 0.
        observations = []
        val_labels = []
        val_preds = []

        self.model.eval()

        pbar = tqdm(self.val_loader, leave=False)
        for i, data in enumerate(pbar):
            keys_input = ['s_input', 'm_input', 'g_input']
            inputs = {key: data[key] for key in keys_input}
            mask = data['target']

            outputs = self.model(inputs)

            loss = self.criterion(outputs, mask)
            val_loss += loss.item()
            epoch_loss = val_loss / (i + 1)

            observations += data['observation'].squeeze().tolist()
            val_mask += mask.squeeze().tolist()
            val_preds += outputs.squeeze().tolist()

            # Update the progress bar with new metrics values
            pbar.set_description(f'VAL - Batch: {i + 1}/{len(self.val_loader)} - '
                                 f'Epoch Loss: {epoch_loss:.5f} - '
                                 f'Batch Loss: {loss.item():.5f}')

        val_loss /= len(self.val_loader)
        precision, recall, f05_score, _ = precision_recall_fscore_support(val_mask, val_pred, beta=0.5)

        return val_loss, precision, recall, f05_score

    def save(self, score: float):
        """ Save the model if it is the better than the previous sevaed one.

        :param score: current model epoch score
        :type score: float
        """
        save_folder = join(ROOT_DIR, 'models')

        if score > self.val_best_r2_score:
            self.val_best_r2_score = score
            os.makedirs(save_folder, exist_ok=True)

            # delete the former best model
            former_model = [f for f in os.listdir(save_folder) if f.split('_')[-1] == f'{self.timestamp}.pt']
            if len(former_model) == 1:
                os.remove(join(save_folder, former_model[0]))

            # save the new model
            score = str(score)[:7].replace('.', '-')
            file_name = f'{score}_model_{self.timestamp}.pt'
            save_path = join(save_folder, file_name)
            torch.save(self.model, save_path)

    def train(self):
        """ Main function to train the model. """
        iter_epoch = tqdm(range(self.epochs), leave=False)

        for epoch in iter_epoch:
            iter_epoch.set_description(f'EPOCH {epoch + 1}/{self.epochs}')
            train_loss = self.train_one_epoch()

            val_loss, val_r2_score, val_mean_r2_score = self.val_one_epoch()
            self.scheduler.step(val_loss)
            self.save(val_mean_r2_score)

            # log the metrics to W&B
            wandb.log({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_r2_score': val_r2_score,
                'val_mean_r2_score': val_mean_r2_score,
                'val_best_r2_score': self.val_best_r2_score
            })

            # Write the finished epoch metrics values
            iter_epoch.write(f'EPOCH {epoch + 1}/{self.epochs}: '
                             f'Train = {train_loss:.5f} - '
                             f'Val = {val_loss:.5f} - '
                             f'Val R2 = {val_r2_score:.5f} - '
                             f'Val mean R2 = {val_mean_r2_score:.5f}')


# # Create an instance of the combined loss function
# alpha = 0.7  # Weight for BCE loss
# beta = 0.3  # Weight for Dice loss
# loss_function = CombinedLoss(alpha, beta)