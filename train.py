import os
import argparse
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, Subset
from dataloader import Datasetloader_, train_split
from model import BoundaryAwareAttentionUNet
from loss import CombinedLoss
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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

# Validatation

def validate(model, val_loader, criterion, device, threshold=0.5):
    model.eval()
    val_loss = 0.0
    dice_running = 0.0

    iou_list, dice_list, acc_list, prec_list, rec_list, f1_list = [], [], [], [], [], []

    with torch.no_grad():
        for inputs, masks in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = criterion(outputs, masks)
            val_loss += loss.item() * inputs.size(0)

            pred = (torch.sigmoid(outputs) > threshold).float()
            iou_list.append(compute_iou(pred, masks))
            d = compute_dice(pred, masks)
            dice_list.append(d)

            acc, precision, recall, f1 = compute_metrics(pred.int(), masks.int())
            acc_list.append(acc); prec_list.append(precision); rec_list.append(recall); f1_list.append(f1)

            intersection = (pred * masks.float()).sum()
            dice_running += (2.0 * intersection / (pred.sum() + masks.float().sum() + 1e-8)).item() * inputs.size(0)

    n = len(val_loader.dataset)
    return {
        "val_loss": val_loss / n,
        "val_dice": dice_running / n,
        "avg_iou": sum(iou_list) / max(len(iou_list), 1),
        "avg_dice": sum(dice_list) / max(len(dice_list), 1),
        "avg_accuracy": sum(acc_list) / max(len(acc_list), 1),
        "avg_precision": sum(prec_list) / max(len(prec_list), 1),
        "avg_recall": sum(rec_list) / max(len(rec_list), 1),
        "avg_f1": sum(f1_list) / max(len(f1_list), 1),
    }


# Train

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    full_dataset = Datasetloader_(
        root_dir=args.data_root,
        transform=None  
    )

    # Split into train/test indices
    train_dataset, test_dataset = train_split(full_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=8)
    val_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=8)
    model = BoundaryAwareAttentionUNet(num_classes=1, pretrained=True).to(device)
    criterion = CombinedLoss(dice_weight=args.dice_weight, bce_weight=args.bce_weight, edge_weight=args.edge_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    best_val = float('inf')

    for epoch in range(args.epochs):
        model.train()
        running = 0.0

        for inputs, masks in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs)
            loss = criterion(outputs, masks)
            loss.backward()

            optimizer.step()

            running += loss.item() * inputs.size(0)

        train_loss = running / len(train_loader.dataset)

        metrics = validate(model, val_loader, criterion, device, threshold=0.5)
        scheduler.step(metrics["val_loss"])

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train {train_loss:.4f} | Val {metrics['val_loss']:.4f} | "
              f"Dice {metrics['val_dice']:.4f} | IoU {metrics['avg_iou']:.4f} | "
              f"F1 {metrics['avg_f1']:.4f}")

        if metrics["val_loss"] < best_val:
            best_val = metrics["val_loss"]
            best_path = os.path.join(args.save_dir, "best_boundary_aware_unet.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  -> Saved new best to {best_path}")

    final_path = os.path.join(args.save_dir, "last_boundary_aware_unet.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Training complete. Final model saved to {final_path}")
def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True, help="Path to dataset root containing image/ and masks/")
    p.add_argument("--save_dir", type=str, default="runs/exp1")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dice_weight", type=float, default=1.0)
    p.add_argument("--bce_weight", type=float, default=1.0)
    p.add_argument("--edge_weight", type=float, default=0.5)
    return p

if __name__ == "__main__":
    args = get_parser().parse_args()
    train(args)