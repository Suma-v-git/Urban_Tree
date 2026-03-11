#!/usr/bin/env python3
"""
Urban Tree Segmentation Project Setup Script
This script sets up the entire project structure and provides guidance for getting started.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def create_project_structure():
    """Create the complete project structure"""
    
    print("Creating Urban Tree Segmentation Project Structure...")
    
    # Define directories
    directories = [
        'data/raw/images',
        'data/raw/masks', 
        'data/processed/train/images',
        'data/processed/train/masks',
        'data/processed/val/images',
        'data/processed/val/masks',
        'data/processed/test/images',
        'data/processed/test/masks',
        'src/data',
        'src/models',
        'src/training',
        'src/inference',
        'src/utils',
        'src/web',
        'app/static',
        'app/templates',
        'models',
        'notebooks',
        'logs',
        'checkpoints',
        'inference_results',
        'examples'
    ]
    
    # Create directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  Created: {directory}")
    
    print("Project structure created successfully!")

def install_dependencies():
    """Install required dependencies"""
    
    print("\nInstalling dependencies...")
    
    try:
        # Install requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False
    
    return True

def create_example_notebooks():
    """Create example Jupyter notebooks"""
    
    print("\nCreating example notebooks...")
    
    # Dataset exploration notebook
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Urban Tree Segmentation - Dataset Exploration\n",
                    "\n",
                    "This notebook helps you explore and understand your tree segmentation dataset."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import sys\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import cv2\n",
                    "from PIL import Image\n",
                    "\n",
                    "# Add src to path\n",
                    "sys.path.append('../src')\n",
                    "\n",
                    "from src.data.dataset import explore_dataset, TreeSegmentationDataset\n",
                    "from src.data.prepare_data import analyze_dataset_statistics"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Dataset Overview"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Update these paths according to your dataset location\n",
                    "image_dir = '../data/raw/images'\n",
                    "mask_dir = '../data/raw/masks'\n",
                    "\n",
                    "# Explore dataset\n",
                    "explore_dataset(image_dir, mask_dir, num_samples=5)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Dataset Statistics"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Analyze dataset statistics\n",
                    "stats = analyze_dataset_statistics(image_dir, mask_dir)\n",
                    "print(f\"Total images: {stats['total_images']}\")\n",
                    "print(f\"Average tree coverage: {stats['average_coverage']:.2f}%\")\n",
                    "print(f\"Average trees per image: {stats['average_trees_per_image']:.1f}\")"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Save notebook
    import json
    with open('notebooks/01_dataset_exploration.ipynb', 'w') as f:
        json.dump(notebook_content, f, indent=2)
    
    print("  Created: notebooks/01_dataset_exploration.ipynb")

def create_example_scripts():
    """Create example scripts"""
    
    print("\nCreating example scripts...")
    
    # Quick start script
    quick_start_content = '''#!/usr/bin/env python3
"""
Quick Start Script for Urban Tree Segmentation
This script provides a quick way to get started with the project.
"""

import os
import sys

def main():
    print("Urban Tree Segmentation - Quick Start")
    print("=" * 50)
    
    # Check if model exists
    model_path = "checkpoints/best_model.pth"
    if not os.path.exists(model_path):
        print("Model not found! Please train the model first:")
        print("   python src/training/train.py")
        return
    
    # Check if data exists
    data_dir = "data/processed"
    if not os.path.exists(data_dir):
        print("Processed data not found! Please prepare the dataset first:")
        print("   python src/data/prepare_data.py")
        return
    
    print("Setup complete! You can now:")
    print("   1. Run training: python src/training/train.py")
    print("   2. Test inference: python src/inference/predict.py --image_path path/to/image.jpg")
    print("   3. Start web app: streamlit run app/app.py")
    print("   4. Explore notebooks: jupyter notebook notebooks/")

if __name__ == "__main__":
    main()
'''
    
    with open('quick_start.py', 'w') as f:
        f.write(quick_start_content)
    
    print("  Created: quick_start.py")

def create_documentation():
    """Create additional documentation"""
    
    print("\nCreating documentation...")
    
    # Usage guide
    usage_guide = '''# Urban Tree Segmentation - Usage Guide

## Getting Started

### 1. Dataset Preparation
1. Download your dataset from Kaggle
2. Place images in `data/raw/images/`
3. Place masks in `data/raw/masks/`
4. Run: `python src/data/prepare_data.py`

### 2. Model Training
1. Ensure dataset is prepared
2. Run: `python src/training/train.py`
3. Monitor training progress in `logs/`

### 3. Inference
1. Single image: `python src/inference/predict.py --image_path path/to/image.jpg`
2. Batch processing: `python src/inference/predict.py --image_dir path/to/images --batch`
3. City analysis: `python src/inference/predict.py --image_dir path/to/images --batch --city_name "City Name"`

### 4. Web Application
1. Start Streamlit app: `streamlit run app/app.py`
2. Open browser to http://localhost:8501
3. Upload images and get instant analysis

## Features

- **Tree Detection**: Automatic identification of trees in satellite imagery
- **Tree Counting**: Individual tree identification and counting
- **Coverage Analysis**: Calculate tree coverage percentage
- **Spatial Analysis**: Tree location and distribution analysis
- **Interactive Visualization**: Web-based interface for easy use
- **Export Options**: Download results in multiple formats

## Model Architecture

- **Backbone**: ResNet-34 (pre-trained on ImageNet)
- **Segmentation**: U-Net architecture
- **Input Size**: 256x256 pixels
- **Output**: Binary segmentation mask (tree vs background)

## Performance Metrics

- **IoU (Intersection over Union)**: Measures segmentation accuracy
- **Dice Coefficient**: Similar to IoU, penalizes false positives/negatives
- **Pixel Accuracy**: Overall pixel-level accuracy
- **Tree Count Accuracy**: Accuracy of individual tree detection

## Tips for Best Results

1. **High-Quality Images**: Use clear, high-resolution satellite imagery
2. **Proper Preprocessing**: Ensure images are properly normalized
3. **Balanced Dataset**: Maintain good balance between tree and non-tree pixels
4. **Data Augmentation**: Use various augmentation techniques during training
5. **Regular Validation**: Monitor validation metrics to avoid overfitting

## Troubleshooting

### Common Issues

1. **Model Not Found**: Ensure you've trained the model first
2. **Memory Issues**: Reduce batch size or image resolution
3. **Poor Results**: Check dataset quality and preprocessing
4. **Slow Training**: Use GPU acceleration if available

### Getting Help

- Check the logs directory for detailed error messages
- Review training metrics in `logs/training_history.png`
- Use the exploration notebooks to understand your data
'''
    
    with open('USAGE_GUIDE.md', 'w') as f:
        f.write(usage_guide)
    
    print("  Created: USAGE_GUIDE.md")

def main():
    """Main setup function"""
    
    print("Urban Tree Segmentation Project Setup")
    print("=" * 50)
    
    # Create project structure
    create_project_structure()
    
    # Install dependencies
    if not install_dependencies():
        print("Setup failed during dependency installation")
        return
    
    # Create example notebooks
    create_example_notebooks()
    
    # Create example scripts
    create_example_scripts()
    
    # Create documentation
    create_documentation()
    
    print("\nSetup completed successfully!")
    print("\nNext Steps:")
    print("1. Place your dataset in data/raw/")
    print("2. Run: python src/data/prepare_data.py")
    print("3. Run: python src/training/train.py")
    print("4. Run: streamlit run app/app.py")
    print("\nFor detailed instructions, see USAGE_GUIDE.md")

if __name__ == "__main__":
    main()
