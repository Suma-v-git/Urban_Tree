import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

class TreeSegmentationDataset(Dataset):
    """Dataset for tree segmentation from satellite imagery"""
    
    def __init__(self, image_dir, mask_dir, transform=None, image_size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_size = image_size
        
        # Get all image files
        self.images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.masks = [f.replace('.jpg', '.png').replace('.jpeg', '.png') for f in self.images]
        
        print(f"Found {len(self.images)} images and {len(self.masks)} masks")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize
        image = cv2.resize(image, self.image_size)
        mask = cv2.resize(mask, self.image_size)
        
        # Normalize mask to binary (0: background, 1: tree)
        mask = (mask > 127).astype(np.float32)
        
        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask.unsqueeze(0)  # Add channel dimension to mask

def get_transforms(train=True):
    """Get data augmentation transforms"""
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

def create_data_loaders(train_image_dir, train_mask_dir, val_image_dir=None, val_mask_dir=None, 
                       batch_size=16, image_size=(256, 256)):
    """Create training and validation data loaders"""
    
    # Training dataset
    train_dataset = TreeSegmentationDataset(
        train_image_dir, train_mask_dir, 
        transform=get_transforms(train=True),
        image_size=image_size
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    
    # Validation dataset
    val_loader = None
    if val_image_dir and val_mask_dir:
        val_dataset = TreeSegmentationDataset(
            val_image_dir, val_mask_dir,
            transform=get_transforms(train=False),
            image_size=image_size
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )
    
    return train_loader, val_loader

def explore_dataset(image_dir, mask_dir, num_samples=5):
    """Explore and visualize dataset samples"""
    images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))][:num_samples]
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    for i, img_name in enumerate(images):
        # Load image
        img_path = os.path.join(image_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_name = img_name.replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = os.path.join(mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Create overlay
        overlay = image.copy()
        overlay[mask > 127] = [0, 255, 0]  # Green overlay for trees
        
        # Plot
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f'Original Image: {img_name}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title('Tree Mask')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title('Tree Overlay')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('data_exploration.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Example usage
    explore_dataset("data/raw/images", "data/raw/masks")
