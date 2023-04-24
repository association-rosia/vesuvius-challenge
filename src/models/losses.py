import torch.nn as nn
from torchgeometry.losses.dice import DiceLoss

class CombinedLoss(nn.Module):
    def __init__(self, bce_weight, dice_weight):
        super(CombinedLoss, self).__init__()
        self.alpha = bce_weight
        self.beta = dice_weight
        self.bce_loss = nn.BCELoss()
        self.dice_loss = DiceLoss()
        
    def forward(self, input, target):
        # Compute binary cross-entropy (BCE) loss
        bce_loss = self.bce_loss(input, target)
        
        # Compute Dice loss
        dice_loss = self.dice_loss(input, target)
        
        # Combine the losses using weighted sum
        combined_loss = self.alpha * bce_loss + self.beta * dice_loss
        
        return combined_loss