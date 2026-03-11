import torch
import numpy as np
from sklearn.metrics import confusion_matrix

class IoU:
    """Intersection over Union metric"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.intersection = 0
        self.union = 0
    
    def update(self, pred, target):
        """Update IoU with batch predictions"""
        pred = pred.view(-1).cpu().numpy()
        target = target.view(-1).cpu().numpy()
        
        self.intersection += np.sum((pred == 1) & (target == 1))
        self.union += np.sum(((pred == 1) | (target == 1)))
    
    def compute(self):
        """Compute IoU"""
        if self.union == 0:
            return 0.0
        return self.intersection / self.union
    
    def __call__(self, pred, target):
        """Direct computation for single batch"""
        pred = pred.view(-1).cpu().numpy()
        target = target.view(-1).cpu().numpy()
        
        intersection = np.sum((pred == 1) & (target == 1))
        union = np.sum(((pred == 1) | (target == 1)))
        
        if union == 0:
            return torch.tensor(0.0)
        return torch.tensor(intersection / union)

class DiceCoefficient:
    """Dice Coefficient metric"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.intersection = 0
        self.dice_sum = 0
    
    def update(self, pred, target):
        """Update Dice with batch predictions"""
        pred = pred.view(-1).cpu().numpy()
        target = target.view(-1).cpu().numpy()
        
        intersection = np.sum((pred == 1) & (target == 1))
        dice = (2.0 * intersection) / (np.sum(pred) + np.sum(target) + 1e-8)
        
        self.intersection += intersection
        self.dice_sum += dice
    
    def compute(self):
        """Compute average Dice"""
        return self.dice_sum / max(len(self.dice_sum), 1)
    
    def __call__(self, pred, target):
        """Direct computation for single batch"""
        pred = pred.view(-1).cpu().numpy()
        target = target.view(-1).cpu().numpy()
        
        intersection = np.sum((pred == 1) & (target == 1))
        dice = (2.0 * intersection) / (np.sum(pred) + np.sum(target) + 1e-8)
        
        return torch.tensor(dice)

class PixelAccuracy:
    """Pixel accuracy metric"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.correct = 0
        self.total = 0
    
    def update(self, pred, target):
        """Update accuracy with batch predictions"""
        pred = pred.view(-1).cpu().numpy()
        target = target.view(-1).cpu().numpy()
        
        self.correct += np.sum(pred == target)
        self.total += pred.size
    
    def compute(self):
        """Compute accuracy"""
        if self.total == 0:
            return 0.0
        return self.correct / self.total
    
    def __call__(self, pred, target):
        """Direct computation for single batch"""
        pred = pred.view(-1).cpu().numpy()
        target = target.view(-1).cpu().numpy()
        
        correct = np.sum(pred == target)
        total = pred.size
        
        return torch.tensor(correct / total)

class TreeCounter:
    """Tree counting metric using connected components"""
    
    def __init__(self, min_area=10):
        self.min_area = min_area
    
    def count_trees(self, mask):
        """Count individual trees in binary mask"""
        import cv2
        
        # Convert to uint8
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_uint8, connectivity=8
        )
        
        # Filter by minimum area (exclude background)
        tree_count = 0
        for i in range(1, num_labels):  # Skip background (label 0)
            if stats[i, cv2.CC_STAT_AREA] >= self.min_area:
                tree_count += 1
        
        return tree_count, labels, centroids
    
    def __call__(self, pred_mask, true_mask):
        """Compare predicted vs true tree count"""
        pred_count, _, _ = self.count_trees(pred_mask)
        true_count, _, _ = self.count_trees(true_mask)
        
        return {
            'predicted_count': pred_count,
            'true_count': true_count,
            'count_error': abs(pred_count - true_count),
            'count_accuracy': 1.0 - (abs(pred_count - true_count) / max(true_count, 1))
        }

class CoverageAnalyzer:
    """Analyze tree coverage percentage"""
    
    def __init__(self):
        pass
    
    def calculate_coverage(self, mask):
        """Calculate tree coverage percentage"""
        total_pixels = mask.size
        tree_pixels = np.sum(mask > 0)
        coverage_percentage = (tree_pixels / total_pixels) * 100
        
        return {
            'total_pixels': total_pixels,
            'tree_pixels': tree_pixels,
            'coverage_percentage': coverage_percentage
        }
    
    def __call__(self, pred_mask, true_mask):
        """Compare predicted vs true coverage"""
        pred_coverage = self.calculate_coverage(pred_mask)
        true_coverage = self.calculate_coverage(true_mask)
        
        return {
            'predicted_coverage': pred_coverage,
            'true_coverage': true_coverage,
            'coverage_error': abs(pred_coverage['coverage_percentage'] - true_coverage['coverage_percentage'])
        }

def calculate_metrics_batch(predictions, targets):
    """Calculate multiple metrics for a batch"""
    iou_metric = IoU()
    dice_metric = DiceCoefficient()
    accuracy_metric = PixelAccuracy()
    tree_counter = TreeCounter()
    coverage_analyzer = CoverageAnalyzer()
    
    batch_metrics = {
        'iou': [],
        'dice': [],
        'accuracy': [],
        'tree_count': [],
        'coverage': []
    }
    
    for pred, target in zip(predictions, targets):
        # Convert to numpy
        pred_np = pred.cpu().numpy()
        target_np = target.cpu().numpy()
        
        # Basic metrics
        batch_metrics['iou'].append(iou_metric(pred, target).item())
        batch_metrics['dice'].append(dice_metric(pred, target).item())
        batch_metrics['accuracy'].append(accuracy_metric(pred, target).item())
        
        # Tree counting and coverage
        batch_metrics['tree_count'].append(tree_counter(pred_np, target_np))
        batch_metrics['coverage'].append(coverage_analyzer(pred_np, target_np))
    
    # Calculate averages
    avg_metrics = {
        'avg_iou': np.mean(batch_metrics['iou']),
        'avg_dice': np.mean(batch_metrics['dice']),
        'avg_accuracy': np.mean(batch_metrics['accuracy']),
        'avg_tree_count_accuracy': np.mean([m['count_accuracy'] for m in batch_metrics['tree_count']]),
        'avg_coverage_error': np.mean([m['coverage_error'] for m in batch_metrics['coverage']])
    }
    
    return avg_metrics, batch_metrics

if __name__ == "__main__":
    # Test metrics
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy predictions and targets
    batch_size = 4
    height, width = 256, 256
    
    predictions = torch.rand(batch_size, 1, height, width) > 0.5
    targets = torch.rand(batch_size, 1, height, width) > 0.5
    
    # Calculate metrics
    avg_metrics, batch_metrics = calculate_metrics_batch(predictions, targets)
    
    print("Average Metrics:")
    for key, value in avg_metrics.items():
        print(f"{key}: {value:.4f}")
