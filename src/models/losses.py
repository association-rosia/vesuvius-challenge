import torch
from torch import nn
from torchmetrics.classification import Dice


class BCEDiceWithLogitsLoss(nn.Module):
    def __init__(self, bce_weight, dice_threshold):
        super(BCEDiceWithLogitsLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_threshold = dice_threshold
        self.dice_weight = 1 - bce_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceWithLogitsLoss(threshold=self.dice_threshold)

    def forward(self, predictions, targets):
        bce_loss = self.bce_loss(predictions, targets)
        dice_loss = self.dice_loss(predictions, targets)
        bce_dice_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss

        return bce_dice_loss


class DiceWithLogitsLoss(nn.Module):
    def __init__(self, threshold):
        super(DiceWithLogitsLoss, self).__init__()
        self.threshold = threshold
        self.sigmoid = nn.Sigmoid()
        self.dice = Dice(threshold=self.threshold)

    def forward(self, preds, target):
        target = target.to(dtype=torch.uint8)
        preds = self.sigmoid(preds)
        loss = 1 - self.dice(preds, target)
        
        return loss
