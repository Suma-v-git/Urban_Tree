# Urban Tree Segmentation Project

## 🌳 Live Demo
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://urban-tree-segmentation.streamlit.app)

## Project Overview
This project uses deep learning to segment trees from satellite imagery, providing quantitative analysis for urban planning and environmental monitoring.

## Features
- **Semantic Segmentation**: U-Net based model for tree detection
- **Tree Counting**: Individual tree identification and counting
- **Coverage Analysis**: Calculate tree coverage percentage
- **Interactive Web App**: Upload images and get instant analysis
- **City-specific Analysis**: Filter results by city/location
- **Visualization**: Tree masks, overlays, and statistical reports

## Project Structure
```
Urban_Tree_Segmentation/
├── data/
│   ├── raw/                 # Original dataset
│   └── processed/          # Preprocessed data
├── src/
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Model architectures
│   ├── training/          # Training scripts
│   ├── inference/         # Inference and prediction
│   ├── utils/             # Utility functions
│   └── web/               # Web application
├── app/
│   ├── static/            # Static files (CSS, JS)
│   └── templates/         # HTML templates
├── models/                # Trained models
├── notebooks/             # Jupyter notebooks
├── logs/                  # Training logs
├── checkpoints/           # Model checkpoints
└── requirements.txt       # Dependencies
```

## Installation
```bash
pip install -r requirements.txt
```

## Usage
1. **Training**: `python src/training/train.py`
2. **Inference**: `python src/inference/predict.py --image_path path/to/image`
3. **Web App**: `streamlit run app/app.py`

## Dataset
Download from Kaggle: [Urban Tree Segmentation Dataset]
Place in `data/raw/` directory

## Model Architecture
- **Backbone**: ResNet-34
- **Segmentation**: U-Net
- **Input Size**: 256x256
- **Classes**: Background (0), Tree (1)

## Metrics
- **IoU (Intersection over Union)**
- **Dice Coefficient**
- **Pixel Accuracy**
- **Tree Count Accuracy**
