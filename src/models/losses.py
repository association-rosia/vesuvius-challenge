import torch.nn as nn
from torchgeometry.losses.dice import DiceLoss

class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=None, dice_weight=1):
        super(CombinedLoss, self).__init__()

        if not ((dice_weight is None) ^ (bce_weight is None)):
            raise TypeError(f'dice_weight and bce_weight are both None or both defined. dice_weight={type(dice_weight)}, bce_weight={type(bce_weight)}')
        if not dice_weight is None:
            self.dice_weight = dice_weight
            self.bce_weight = 1 - dice_weight
        else:
            self.bce_weight = bce_weight
            self.dice_weight = 1 - bce_weight
            
        self.bce_loss = nn.BCELoss()
        self.dice_loss = DiceLoss()
        
    def forward(self, inputs, targets):
        # Compute binary cross-entropy (BCE) loss
        bce_loss = self.bce_loss(inputs, targets)
        
        # Compute Dice loss
        dice_loss = self.dice_loss(inputs, targets)
        
        # Combine the losses using weighted sum
        combined_loss = self.alpha * bce_loss + self.beta * dice_loss
        
        return combined_loss
    
    def half(self):
        self.bce_loss = self.bce_loss.half()
        self.dice_loss = self.dice_loss.half()