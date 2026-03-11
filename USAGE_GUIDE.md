# Urban Tree Segmentation - Usage Guide

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
