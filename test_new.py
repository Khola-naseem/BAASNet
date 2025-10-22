import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import os
import argparse
import pandas as pd
import numpy as np
import tarfile
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from dataloader import Datasetloader_
from model import BoundaryAwareAttentionUNet

def compute_iou(pred, target):
    intersection = torch.logical_and(pred, target).float().sum((1, 2))
    union = torch.logical_or(pred, target).float().sum((1, 2))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()

def compute_dice(pred, target):
    intersection = torch.logical_and(pred, target).float().sum((1, 2))
    dice = (2. * intersection + 1e-6) / (pred.float().sum((1, 2)) + target.float().sum((1, 2)) + 1e-6)
    return dice.mean().item()

def compute_metrics(pred, target):
    pred_flat = pred.view(-1).cpu().numpy()
    target_flat = target.view(-1).cpu().numpy()

    acc = accuracy_score(target_flat, pred_flat)
    precision = precision_score(target_flat, pred_flat, zero_division=0)
    recall = recall_score(target_flat, pred_flat, zero_division=0)
    f1 = f1_score(target_flat, pred_flat, zero_division=0)

    return acc, precision, recall, f1
@torch.no_grad()
def evaluate_with_metrics(model, loader, device):
    model.eval()

    all_iou, all_dice, all_acc, all_prec, all_rec, all_f1 = [], [], [], [], [], []

    for inputs, masks in loader:
        inputs = inputs.to(device, non_blocking=True)        
        masks  = masks.to(device, non_blocking=True)            
        outputs = model(inputs)                                
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        preds =  (torch.sigmoid(outputs) > 0.5).float()  
        masks_ = (masks > 0.5).float()              
        iou  = compute_iou(preds, masks_)
        dice = compute_dice(preds, masks_)
        acc, prec, rec, f1 = compute_metrics(preds, masks_)

        all_iou.append(iou)
        all_dice.append(dice)
        all_acc.append(acc)
        all_prec.append(prec)
        all_rec.append(rec)
        all_f1.append(f1)

    print("\n=== Test Results ===")
    print(f"IoU:       {np.mean(all_iou):.4f}")
    print(f"Dice:      {np.mean(all_dice):.4f}")
    print(f"Accuracy:  {np.mean(all_acc):.4f}")
    print(f"Precision: {np.mean(all_prec):.4f}")
    print(f"Recall:    {np.mean(all_rec):.4f}")
    print(f"F1 Score:  {np.mean(all_f1):.4f}")
    

def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True, help="Path to dataset root containing image/")
    p.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint .pth file")
    p.add_argument("--batch_size", type=int, default=32, help="batch_size")
    p.add_argument("--seed", type=int, default=42)
    return p
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BoundaryAwareAttentionUNet(num_classes=1).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    val_transform = A.Compose([
    A.Resize(256, 256),
    ToTensorV2()
])
    test_dataset = Datasetloader_(root_dir=args.data_path, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    evaluate_with_metrics(model, test_loader, device=device)

if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)

