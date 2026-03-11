import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import create_data_loaders
from src.models.unet import get_model
from src.utils.metrics import IoU, DiceCoefficient, PixelAccuracy
from src.utils.utils import save_checkpoint, load_checkpoint

class TreeSegmentationTrainer:
    """Trainer class for tree segmentation"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = get_model(
            model_type=config['model_type'],
            encoder_name=config.get('encoder_name', 'resnet34'),
            pretrained=config.get('pretrained', True)
        )
        self.model.to(self.device)
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Metrics
        self.iou_metric = IoU()
        self.dice_metric = DiceCoefficient()
        self.accuracy_metric = PixelAccuracy()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_ious = []
        self.val_ious = []
        self.train_dices = []
        self.val_dices = []
        
        # Create directories
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_iou = 0
        total_dice = 0
        total_accuracy = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (images, masks) in enumerate(pbar):
            images, masks = images.to(self.device), masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                predictions = torch.sigmoid(outputs) > 0.5
                iou = self.iou_metric(predictions, masks)
                dice = self.dice_metric(predictions, masks)
                accuracy = self.accuracy_metric(predictions, masks)
            
            total_loss += loss.item()
            total_iou += iou.item()
            total_dice += dice.item()
            total_accuracy += accuracy.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'IoU': f'{iou.item():.4f}',
                'Dice': f'{dice.item():.4f}'
            })
        
        return {
            'loss': total_loss / len(train_loader),
            'iou': total_iou / len(train_loader),
            'dice': total_dice / len(train_loader),
            'accuracy': total_accuracy / len(train_loader)
        }
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        total_iou = 0
        total_dice = 0
        total_accuracy = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for images, masks in pbar:
                images, masks = images.to(self.device), masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                predictions = torch.sigmoid(outputs) > 0.5
                iou = self.iou_metric(predictions, masks)
                dice = self.dice_metric(predictions, masks)
                accuracy = self.accuracy_metric(predictions, masks)
                
                total_loss += loss.item()
                total_iou += iou.item()
                total_dice += dice.item()
                total_accuracy += accuracy.item()
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'IoU': f'{iou.item():.4f}',
                    'Dice': f'{dice.item():.4f}'
                })
        
        return {
            'loss': total_loss / len(val_loader),
            'iou': total_iou / len(val_loader),
            'dice': total_dice / len(val_loader),
            'accuracy': total_accuracy / len(val_loader)
        }
    
    def train(self, train_loader, val_loader=None, num_epochs=100):
        """Main training loop"""
        best_val_loss = float('inf')
        best_val_iou = 0
        
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            self.train_losses.append(train_metrics['loss'])
            self.train_ious.append(train_metrics['iou'])
            self.train_dices.append(train_metrics['dice'])
            
            # Validation
            if val_loader:
                val_metrics = self.validate_epoch(val_loader)
                self.val_losses.append(val_metrics['loss'])
                self.val_ious.append(val_metrics['iou'])
                self.val_dices.append(val_metrics['dice'])
                
                # Update scheduler
                self.scheduler.step(val_metrics['loss'])
                
                # Save best model
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    best_val_iou = val_metrics['iou']
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'val_loss': best_val_loss,
                        'val_iou': best_val_iou,
                        'config': self.config
                    }, f'checkpoints/best_model.pth')
                
                # Print epoch summary
                print(f"Train Loss: {train_metrics['loss']:.4f}, IoU: {train_metrics['iou']:.4f}")
                print(f"Val Loss: {val_metrics['loss']:.4f}, IoU: {val_metrics['iou']:.4f}")
                print(f"Best Val IoU: {best_val_iou:.4f}")
            else:
                print(f"Train Loss: {train_metrics['loss']:.4f}, IoU: {train_metrics['iou']:.4f}")
        
        # Save final model
        save_checkpoint({
            'epoch': num_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, f'checkpoints/final_model.pth')
        
        # Plot training history
        self.plot_training_history()
        
        print("\nTraining completed!")
        print(f"Best validation IoU: {best_val_iou:.4f}")
    
    def plot_training_history(self):
        """Plot training and validation metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        if self.val_losses:
            axes[0, 0].plot(self.val_losses, label='Val Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        
        # IoU
        axes[0, 1].plot(self.train_ious, label='Train IoU')
        if self.val_ious:
            axes[0, 1].plot(self.val_ious, label='Val IoU')
        axes[0, 1].set_title('IoU')
        axes[0, 1].legend()
        
        # Dice
        axes[1, 0].plot(self.train_dices, label='Train Dice')
        if self.val_dices:
            axes[1, 0].plot(self.val_dices, label='Val Dice')
        axes[1, 0].set_title('Dice Coefficient')
        axes[1, 0].legend()
        
        # Learning rate
        axes[1, 1].plot([param_group['lr'] for param_group in self.optimizer.param_groups])
        axes[1, 1].set_title('Learning Rate')
        
        plt.tight_layout()
        plt.savefig('logs/training_history.png', dpi=150, bbox_inches='tight')
        plt.show()

def main():
    """Main training function"""
    # Configuration
    config = {
        'model_type': 'pretrained',
        'encoder_name': 'resnet34',
        'pretrained': True,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'batch_size': 16,
        'num_epochs': 100,
        'image_size': (256, 256)
    }
    
    # Save config
    with open('logs/config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_image_dir='data/processed/train/images',
        train_mask_dir='data/processed/train/masks',
        val_image_dir='data/processed/val/images',
        val_mask_dir='data/processed/val/masks',
        batch_size=config['batch_size'],
        image_size=config['image_size']
    )
    
    # Initialize trainer
    trainer = TreeSegmentationTrainer(config)
    
    # Start training
    trainer.train(train_loader, val_loader, config['num_epochs'])

if __name__ == "__main__":
    main()
