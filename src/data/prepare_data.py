import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def prepare_dataset_structure(raw_data_dir, processed_data_dir, val_split=0.2, test_split=0.1):
    """Prepare dataset structure for training"""
    
    print("Preparing dataset structure...")
    
    # Create directories
    train_dir = os.path.join(processed_data_dir, 'train')
    val_dir = os.path.join(processed_data_dir, 'val')
    test_dir = os.path.join(processed_data_dir, 'test')
    
    for dir_path in [train_dir, val_dir, test_dir]:
        os.makedirs(os.path.join(dir_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dir_path, 'masks'), exist_ok=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend([f for f in os.listdir(raw_data_dir) if f.lower().endswith(ext)])
    
    print(f"Found {len(image_files)} images")
    
    # Split dataset
    train_files, temp_files = train_test_split(image_files, test_size=(val_split + test_split), random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=(test_split / (val_split + test_split)), random_state=42)
    
    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    
    # Copy files to appropriate directories
    def copy_files(file_list, target_dir):
        for file_name in file_list:
            # Copy image
            src_path = os.path.join(raw_data_dir, file_name)
            dst_path = os.path.join(target_dir, 'images', file_name)
            shutil.copy2(src_path, dst_path)
            
            # Copy corresponding mask (assuming same name with .png extension)
            mask_name = os.path.splitext(file_name)[0] + '.png'
            src_mask_path = os.path.join(raw_data_dir.replace('images', 'masks'), mask_name)
            dst_mask_path = os.path.join(target_dir, 'masks', mask_name)
            
            if os.path.exists(src_mask_path):
                shutil.copy2(src_mask_path, dst_mask_path)
            else:
                print(f"Warning: Mask not found for {file_name}")
    
    copy_files(train_files, train_dir)
    copy_files(val_files, val_dir)
    copy_files(test_files, test_dir)
    
    print("Dataset structure prepared successfully!")
    
    return train_dir, val_dir, test_dir

def analyze_dataset_statistics(image_dir, mask_dir):
    """Analyze dataset statistics"""
    
    print("Analyzing dataset statistics...")
    
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    total_pixels = 0
    tree_pixels = 0
    image_sizes = []
    tree_counts = []
    
    for img_file in image_files:
        # Load image
        img_path = os.path.join(image_dir, img_file)
        image = cv2.imread(img_path)
        image_sizes.append(image.shape[:2])
        total_pixels += image.shape[0] * image.shape[1]
        
        # Load mask
        mask_name = os.path.splitext(img_file)[0] + '.png'
        mask_path = os.path.join(mask_dir, mask_name)
        
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            tree_pixels += np.sum(mask > 127)
            
            # Count individual trees
            num_labels, _, _, _ = cv2.connectedComponentsWithStats((mask > 127).astype(np.uint8))
            tree_counts.append(num_labels - 1)  # Exclude background
    
    # Calculate statistics
    avg_coverage = (tree_pixels / total_pixels) * 100
    avg_image_size = np.mean(image_sizes, axis=0)
    avg_tree_count = np.mean(tree_counts)
    
    stats = {
        'total_images': len(image_files),
        'average_coverage': avg_coverage,
        'average_image_size': avg_image_size.tolist(),
        'average_trees_per_image': avg_tree_count,
        'total_tree_pixels': tree_pixels,
        'total_pixels': total_pixels
    }
    
    print(f"Dataset Statistics:")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Average tree coverage: {stats['average_coverage']:.2f}%")
    print(f"  Average image size: {stats['average_image_size'][0]}x{stats['average_image_size'][1]}")
    print(f"  Average trees per image: {stats['average_trees_per_image']:.1f}")
    
    return stats

def create_dataset_visualization(image_dir, mask_dir, output_path='dataset_analysis.png', num_samples=6):
    """Create visualization of dataset samples"""
    
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    sample_files = np.random.choice(image_files, min(num_samples, len(image_files)), replace=False)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    for i, img_file in enumerate(sample_files):
        # Load image
        img_path = os.path.join(image_dir, img_file)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_name = os.path.splitext(img_file)[0] + '.png'
        mask_path = os.path.join(mask_dir, mask_name)
        
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Create overlay
            overlay = image.copy()
            overlay[mask > 127] = [0, 255, 0]  # Green overlay for trees
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            overlay = image.copy()
        
        # Plot
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f'Image: {img_file}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title('Tree Mask')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title('Tree Overlay')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Dataset visualization saved to {output_path}")

def validate_dataset(image_dir, mask_dir):
    """Validate dataset integrity"""
    
    print("Validating dataset...")
    
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith('.png')]
    
    issues = []
    
    # Check for missing masks
    for img_file in image_files:
        mask_name = os.path.splitext(img_file)[0] + '.png'
        if mask_name not in mask_files:
            issues.append(f"Missing mask for {img_file}")
    
    # Check for corrupted files
    for img_file in image_files:
        try:
            img_path = os.path.join(image_dir, img_file)
            image = cv2.imread(img_path)
            if image is None:
                issues.append(f"Corrupted image: {img_file}")
        except Exception as e:
            issues.append(f"Error loading {img_file}: {str(e)}")
    
    for mask_file in mask_files:
        try:
            mask_path = os.path.join(mask_dir, mask_file)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                issues.append(f"Corrupted mask: {mask_file}")
        except Exception as e:
            issues.append(f"Error loading mask {mask_file}: {str(e)}")
    
    if issues:
        print("Dataset validation issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Dataset validation passed!")
    
    return len(issues) == 0

def main():
    """Main data preparation function"""
    
    # Configuration
    raw_data_dir = "data/raw/images"  # Update with your actual path
    raw_masks_dir = "data/raw/masks"  # Update with your actual path
    processed_data_dir = "data/processed"
    
    # Create processed data directory
    os.makedirs(processed_data_dir, exist_ok=True)
    
    # Step 1: Prepare dataset structure
    train_dir, val_dir, test_dir = prepare_dataset_structure(
        raw_data_dir, processed_data_dir, val_split=0.2, test_split=0.1
    )
    
    # Step 2: Analyze dataset statistics
    stats = analyze_dataset_statistics(train_dir, os.path.join(train_dir, 'masks'))
    
    # Step 3: Create visualization
    create_dataset_visualization(
        train_dir, os.path.join(train_dir, 'masks'), 
        'logs/dataset_analysis.png', num_samples=6
    )
    
    # Step 4: Validate dataset
    is_valid = validate_dataset(train_dir, os.path.join(train_dir, 'masks'))
    
    if is_valid:
        print("\n✅ Dataset preparation completed successfully!")
        print(f"Train directory: {train_dir}")
        print(f"Validation directory: {val_dir}")
        print(f"Test directory: {test_dir}")
    else:
        print("\n❌ Dataset preparation completed with issues. Please check the validation output.")

if __name__ == "__main__":
    main()
