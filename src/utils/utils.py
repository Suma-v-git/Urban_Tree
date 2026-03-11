import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import json
from datetime import datetime

def save_checkpoint(state, filename):
    """Save model checkpoint"""
    print(f"Saving checkpoint to {filename}")
    torch.save(state, filename)

def load_checkpoint(filename, model, optimizer=None, scheduler=None):
    """Load model checkpoint"""
    print(f"Loading checkpoint from {filename}")
    checkpoint = torch.load(filename, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    val_loss = checkpoint.get('val_loss', float('inf'))
    val_iou = checkpoint.get('val_iou', 0.0)
    
    return epoch, val_loss, val_iou

def visualize_prediction(image, mask, prediction, save_path=None):
    """Visualize image, ground truth mask, and prediction"""
    # Convert tensors to numpy if needed
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy().transpose(1, 2, 0)
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy().squeeze()
    if isinstance(prediction, torch.Tensor):
        prediction = torch.sigmoid(prediction).cpu().numpy().squeeze()
        prediction = (prediction > 0.5).astype(np.float32)
    
    # Denormalize image if needed
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Create visualization
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Ground truth mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(prediction, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    # Overlay
    overlay = image.copy()
    green_overlay = np.zeros_like(image)
    green_overlay[:, :, 1] = 255  # Green channel
    
    mask_3d = np.stack([prediction] * 3, axis=-1)
    overlay = overlay * (1 - mask_3d * 0.5) + green_overlay * (mask_3d * 0.5)
    overlay = overlay.astype(np.uint8)
    
    axes[3].imshow(overlay)
    axes[3].set_title('Tree Overlay')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def create_tree_overlay(image, tree_mask, color=[0, 255, 0], alpha=0.5):
    """Create tree overlay on image"""
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy().transpose(1, 2, 0)
    if isinstance(tree_mask, torch.Tensor):
        tree_mask = tree_mask.cpu().numpy().squeeze()
    
    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Create overlay
    overlay = image.copy()
    tree_mask_3d = np.stack([tree_mask] * 3, axis=-1)
    color_array = np.array(color)
    
    overlay = overlay * (1 - tree_mask_3d * alpha) + color_array * (tree_mask_3d * alpha)
    overlay = overlay.astype(np.uint8)
    
    return overlay

def preprocess_image(image_path, target_size=(256, 256)):
    """Preprocess single image for inference"""
    # Read image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Store original size
    original_size = image.shape[:2]
    
    # Resize
    image = cv2.resize(image, target_size)
    
    # Normalize
    image = image.astype(np.float32) / 255.0
    
    # Convert to tensor
    image = torch.from_numpy(image.transpose(2, 0, 1))
    
    return image, original_size

def postprocess_prediction(prediction, original_size):
    """Postprocess prediction to original size"""
    if isinstance(prediction, torch.Tensor):
        prediction = torch.sigmoid(prediction).cpu().numpy().squeeze()
    
    # Binarize
    prediction = (prediction > 0.5).astype(np.uint8)
    
    # Resize to original size
    prediction = cv2.resize(prediction, (original_size[1], original_size[0]))
    
    return prediction

def calculate_tree_statistics(tree_mask, pixel_resolution=1.0):
    """Calculate detailed tree statistics"""
    import cv2
    
    # Convert to uint8
    mask_uint8 = (tree_mask * 255).astype(np.uint8)
    
    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_uint8, connectivity=8
    )
    
    tree_stats = []
    total_area = 0
    
    for i in range(1, num_labels):  # Skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 0:  # Filter out very small components
            total_area += area
            tree_stats.append({
                'tree_id': i,
                'area_pixels': area,
                'area_meters': area * pixel_resolution * pixel_resolution,
                'centroid': centroids[i].tolist(),
                'bbox': [
                    stats[i, cv2.CC_STAT_LEFT],
                    stats[i, cv2.CC_STAT_TOP],
                    stats[i, cv2.CC_STAT_WIDTH],
                    stats[i, cv2.CC_STAT_HEIGHT]
                ]
            })
    
    coverage_percentage = (total_area / (tree_mask.shape[0] * tree_mask.shape[1])) * 100
    
    return {
        'total_trees': len(tree_stats),
        'total_tree_area_pixels': total_area,
        'total_tree_area_meters': total_area * pixel_resolution * pixel_resolution,
        'coverage_percentage': coverage_percentage,
        'tree_details': tree_stats
    }

def save_results(results, output_path):
    """Save analysis results to JSON"""
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return obj
    
    # Deep convert
    import json
    results_json = json.loads(json.dumps(results, default=convert_numpy))
    
    # Add timestamp
    results_json['timestamp'] = datetime.now().isoformat()
    
    with open(output_path, 'w') as f:
        json.dump(results_json, f, indent=2)

def create_report(image_path, results, output_dir):
    """Create visual report with analysis results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load original image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create tree mask overlay
    tree_mask = results['tree_mask']
    overlay = create_tree_overlay(image, tree_mask)
    
    # Create report figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Tree mask
    axes[0, 1].imshow(tree_mask, cmap='gray')
    axes[0, 1].set_title(f'Tree Mask ({results["total_trees"]} trees)')
    axes[0, 1].axis('off')
    
    # Overlay
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title('Tree Overlay')
    axes[1, 0].axis('off')
    
    # Statistics
    stats_text = f"""
    Total Trees: {results['total_trees']}
    Tree Coverage: {results['coverage_percentage']:.2f}%
    Total Tree Area: {results['total_tree_area_meters']:.2f} m²
    
    Average Tree Size: {results['total_tree_area_meters']/max(results['total_trees'], 1):.2f} m²
    """
    
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
    axes[1, 1].set_title('Tree Statistics')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tree_analysis_report.png'), dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    
    # Test visualization
    dummy_image = np.random.rand(256, 256, 3)
    dummy_mask = np.random.rand(256, 256) > 0.5
    dummy_pred = np.random.rand(256, 256) > 0.5
    
    visualize_prediction(dummy_image, dummy_mask, dummy_pred, 'test_visualization.png')
    print("Visualization test completed!")
