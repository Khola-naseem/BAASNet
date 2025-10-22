import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd
import numpy as np
import tarfile
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import BoundaryAwareAttentionUNet
import os
import argparse
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Dataset
class Datasetinfer(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.image_names = [] # To keep track of image names
        for img_name in os.listdir(root_dir):
            img_path = os.path.join(root_dir, img_name)
            if os.path.exists(img_path):
                self.images.append(img_path)
                self.image_names.append(img_name)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image_name = self.image_names[idx]

        image = np.array(Image.open(img_path).convert("RGB"))
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image'].float()


        return {
            'image': image,
            'name': image_name  # Keep this for tracking
        }

# Transform
val_transform = A.Compose([
    A.Resize(256, 256),
    ToTensorV2()
])


@torch.no_grad()
def evaluate_with_metrics(model, loader, device, output_folder, thresh=0.5):
    """
    Saves a binary mask for each image in `output_folder`.

    Args:
        model: torch.nn.Module
        loader: DataLoader that yields dict with keys {'image','mask','name'}
        device: torch.device
        output_folder: str, directory to save results
        thresh: float, threshold on sigmoid probability
        save_prob: bool, also save grayscale probability map [0..255]
        save_preview: bool, also save side-by-side (image | prob | bin mask)
    """
    os.makedirs(output_folder, exist_ok=True)
    model.eval()

    for batch in loader:
        images = batch["image"].to(device)        
        names = batch["name"]                 

        outputs = model(images)                   
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        probs = torch.sigmoid(outputs)           
        preds = (probs > thresh).float()          
        for i, name in enumerate(names):

            pred_bin = preds[i, 0]            
            pred_np = (pred_bin.detach().cpu().numpy() * 255.0).astype(np.uint8)
            name_wo_ext = os.path.splitext(name)[0]
            out_path = os.path.join(output_folder, f"{name_wo_ext}_pred.png")
            cv2.imwrite(out_path, pred_np )

    print(f"Saved binary masks to: {output_folder}")
def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True, help="Path to dataset root containing image/")
    p.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint .pth file")
    p.add_argument("--output_dir", type=str, default="./output")
    p.add_argument("--seed", type=int, default=42)
    return p

def main(agrs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BoundaryAwareAttentionUNet(num_classes=1).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    test_dataset = Datasetinfer(root_dir=args.data_path, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Will save masks to ./output/*.png
    evaluate_with_metrics(
        model,
        test_loader,
        device=device,
        output_folder=args.output_dir,
        thresh=0.5
    )

if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)

