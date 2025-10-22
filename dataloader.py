import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torchvision import models, transforms
class Datasetloader_(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.masks = []
        images_dir = os.path.join(root_dir, 'images')
        masks_dir = os.path.join(root_dir, 'masks')
        image_exts = {'.jpg', '.png'}
        mask_exts  = {'.jpg', '.png'}

        image_basenames = {
            os.path.splitext(f)[0]: f for f in os.listdir(images_dir)
            if os.path.splitext(f)[1].lower() in image_exts
        }
        mask_basenames = {
            os.path.splitext(f)[0]: f for f in os.listdir(masks_dir)
            if os.path.splitext(f)[1].lower() in mask_exts
        }

        for basename, image_file in image_basenames.items():
            if basename in mask_basenames:
                self.images.append(os.path.join(images_dir, image_file))
                self.masks.append(os.path.join(masks_dir, mask_basenames[basename]))


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.images[idx]).convert("RGB"))
        mask = np.array(Image.open(self.masks[idx]).convert("L"))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image'].float()   
            mask = augmented['mask'].unsqueeze(0).float() / 255.0 
        return image, mask
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
train_transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.GaussianBlur(p=0.2),
    A.ColorJitter(p=0.3),
    ToTensorV2()
])
val_transform = A.Compose([
    A.Resize(256, 256),
    ToTensorV2()
])
from torch.utils.data import Subset

class TransformedDataset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        if transform == "train":
            self.transform = train_transform
        else:         
            self.transform = val_transform

    def __getitem__(self, idx):
        img, mask = self.subset[idx]
        augmented = self.transform(image=np.array(img), mask=np.array(mask))

        image = augmented['image'].float()                      # Ensure float32
        mask = augmented['mask'].unsqueeze(0).float() / 255.0   # Ensure [1, H, W] and scaled

        return image, mask

    def __len__(self):
        return len(self.subset)

def train_split(full_dataset):
        # Split into train/test indices
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_indices, test_indices = torch.utils.data.random_split(range(len(full_dataset)), [train_size, test_size])
    train_dataset = TransformedDataset(Subset(full_dataset, train_indices), transform="train")
    test_dataset = TransformedDataset(Subset(full_dataset, test_indices), transform="val")
    return train_dataset, test_dataset