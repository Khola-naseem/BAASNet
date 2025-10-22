import torch
import torch.nn as nn
import torch.nn.functional as F
class BoundaryAwareDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(BoundaryAwareDiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3).to(pred.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3).to(pred.device)
        target_float = target.float()
        target_boundaries_x = F.conv2d(target_float, sobel_x, padding=1)
        target_boundaries_y = F.conv2d(target_float, sobel_y, padding=1)
        target_boundaries = torch.sqrt(target_boundaries_x**2 + target_boundaries_y**2)
        
        # Create boundary weight map (higher weights near boundaries)
        boundary_weight = 1.0 + 5.0 * torch.exp(-((target_boundaries - target_boundaries.max()) ** 2) / 0.1)
        
        # Weighted Dice calculation
        intersection = (pred * target_float * boundary_weight).sum()
        pred_sum = (pred * boundary_weight).sum()
        target_sum = (target_float * boundary_weight).sum()
        
        dice = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        
        return 1.0 - dice

class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=1.0, bce_weight=1.0, edge_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_loss = BoundaryAwareDiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.edge_weight = edge_weight
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        
    def forward(self, outputs, target):
        if isinstance(outputs, tuple):
            main_pred, edge_pred = outputs
            target_float = target.float()
            
            # Extract target edges
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3).to(target.device)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3).to(target.device)
            
            target_boundaries_x = F.conv2d(target_float, sobel_x, padding=1)
            target_boundaries_y = F.conv2d(target_float, sobel_y, padding=1)
            target_boundaries = torch.sqrt(target_boundaries_x**2 + target_boundaries_y**2)
            target_boundaries = (target_boundaries > 0.1).float()
            
            # Calculate losses
            main_dice_loss = self.dice_loss(main_pred, target)
            main_bce_loss = self.bce_loss(main_pred, target_float)
            edge_bce_loss = self.bce_loss(edge_pred, target_boundaries)
            
            
            # Combine all losses
            loss = (self.dice_weight * main_dice_loss + 
                   self.bce_weight * main_bce_loss + self.edge_weight * edge_bce_loss )
            
            return loss
        else:
            # Inference mode
            target_float = target.float()
            dice_loss = self.dice_loss(outputs, target)
            bce_loss = self.bce_loss(outputs, target_float)
            return self.dice_weight * dice_loss + self.bce_weight * bce_loss