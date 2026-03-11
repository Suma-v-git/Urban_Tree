#!/usr/bin/env python3
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
