import torch.nn as nn
import torch

class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=None, dice_weight=None):
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
    
    
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        intersection = torch.logical_and(y_pred.bool(), y_true.bool()).sum()
        union = y_pred.sum() + y_true.sum()
        dice = (2 * intersection.float()) / (union.float() + 1e-7)
        loss = 1 - dice.mean()
        return loss
    

if __name__ == '__main__':
    dice_loss = DiceLoss()
    
    prediction = torch.randint(2, size=(8, 256, 256))
    label = torch.randint(2, size=(8, 256, 256))
    
    print(dice_loss(prediction, label))