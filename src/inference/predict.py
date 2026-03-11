import os
import torch
import argparse
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import json
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.unet import get_model
from src.utils.utils import (
    preprocess_image, postprocess_prediction, 
    visualize_prediction, calculate_tree_statistics,
    save_results, create_report
)
from src.utils.metrics import TreeCounter, CoverageAnalyzer

class TreeSegmentationPredictor:
    """Predictor class for tree segmentation inference"""
    
    def __init__(self, model_path, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = get_model('pretrained')
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize analyzers
        self.tree_counter = TreeCounter()
        self.coverage_analyzer = CoverageAnalyzer()
        
        print(f"Model loaded from {model_path}")
        print(f"Using device: {self.device}")
    
    def predict_single_image(self, image_path, output_dir=None, save_results_flag=True):
        """Predict tree segmentation for a single image"""
        print(f"Processing image: {image_path}")
        
        # Preprocess image
        image_tensor, original_size = preprocess_image(image_path)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            prediction = self.model(image_tensor)
        
        # Postprocess
        tree_mask = postprocess_prediction(prediction[0], original_size)
        
        # Calculate statistics
        tree_stats = calculate_tree_statistics(tree_mask)
        coverage_stats = self.coverage_analyzer(tree_mask)
        
        # Combine results
        results = {
            'image_path': image_path,
            'original_size': list(original_size),
            'tree_mask': tree_mask.tolist(),
            **tree_stats,
            **coverage_stats
        }
        
        # Save results
        if save_results_flag and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save results JSON
            save_results(results, os.path.join(output_dir, 'results.json'))
            
            # Save tree mask
            mask_image = Image.fromarray((tree_mask * 255).astype(np.uint8))
            mask_image.save(os.path.join(output_dir, 'tree_mask.png'))
            
            # Create visualization
            original_image = cv2.imread(image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            visualize_prediction(
                original_image, tree_mask, tree_mask,
                save_path=os.path.join(output_dir, 'visualization.png')
            )
            
            # Create report
            create_report(image_path, results, output_dir)
        
        return results
    
    def predict_batch(self, image_dir, output_dir):
        """Predict tree segmentation for all images in a directory"""
        print(f"Processing batch from directory: {image_dir}")
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend([f for f in os.listdir(image_dir) if f.lower().endswith(ext)])
        
        print(f"Found {len(image_files)} images")
        
        # Process each image
        all_results = []
        for i, image_file in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {image_file}")
            
            image_path = os.path.join(image_dir, image_file)
            image_output_dir = os.path.join(output_dir, os.path.splitext(image_file)[0])
            
            try:
                results = self.predict_single_image(image_path, image_output_dir)
                all_results.append(results)
            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")
                continue
        
        # Save batch summary
        if all_results:
            batch_summary = {
                'total_images': len(image_files),
                'processed_images': len(all_results),
                'total_trees': sum(r['total_trees'] for r in all_results),
                'average_coverage': np.mean([r['coverage_percentage'] for r in all_results]),
                'timestamp': datetime.now().isoformat(),
                'results': all_results
            }
            
            save_results(batch_summary, os.path.join(output_dir, 'batch_summary.json'))
            
            print(f"\nBatch Processing Summary:")
            print(f"Total images: {len(image_files)}")
            print(f"Successfully processed: {len(all_results)}")
            print(f"Total trees detected: {batch_summary['total_trees']}")
            print(f"Average tree coverage: {batch_summary['average_coverage']:.2f}%")
        
        return all_results
    
    def predict_city_analysis(self, image_dir, city_name, output_dir):
        """Perform city-specific tree analysis"""
        print(f"Performing city analysis for: {city_name}")
        
        # Process all images
        city_output_dir = os.path.join(output_dir, city_name.replace(' ', '_'))
        results = self.predict_batch(image_dir, city_output_dir)
        
        if not results:
            print("No images processed successfully")
            return None
        
        # Calculate city-wide statistics
        city_stats = {
            'city_name': city_name,
            'total_images': len(results),
            'total_trees': sum(r['total_trees'] for r in results),
            'total_area_meters': sum(r['total_tree_area_meters'] for r in results),
            'average_coverage': np.mean([r['coverage_percentage'] for r in results]),
            'trees_per_image': np.mean([r['total_trees'] for r in results]),
            'coverage_std': np.std([r['coverage_percentage'] for r in results]),
            'timestamp': datetime.now().isoformat()
        }
        
        # Create city report
        self.create_city_report(city_stats, results, city_output_dir)
        
        return city_stats
    
    def create_city_report(self, city_stats, results, output_dir):
        """Create comprehensive city analysis report"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Tree count distribution
        tree_counts = [r['total_trees'] for r in results]
        axes[0, 0].hist(tree_counts, bins=20, alpha=0.7, color='green')
        axes[0, 0].set_title('Tree Count Distribution')
        axes[0, 0].set_xlabel('Number of Trees')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. Coverage distribution
        coverages = [r['coverage_percentage'] for r in results]
        axes[0, 1].hist(coverages, bins=20, alpha=0.7, color='blue')
        axes[0, 1].set_title('Tree Coverage Distribution')
        axes[0, 1].set_xlabel('Coverage Percentage (%)')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Scatter plot: Trees vs Coverage
        axes[0, 2].scatter(tree_counts, coverages, alpha=0.6, color='red')
        axes[0, 2].set_title('Trees vs Coverage')
        axes[0, 2].set_xlabel('Number of Trees')
        axes[0, 2].set_ylabel('Coverage Percentage (%)')
        
        # 4. City statistics summary
        stats_text = f"""
        City: {city_stats['city_name']}
        
        Total Images: {city_stats['total_images']}
        Total Trees: {city_stats['total_trees']}
        Average Trees/Image: {city_stats['trees_per_image']:.1f}
        
        Average Coverage: {city_stats['average_coverage']:.2f}%
        Coverage Std Dev: {city_stats['coverage_std']:.2f}%
        
        Total Tree Area: {city_stats['total_area_meters']:.2f} m²
        """
        
        axes[1, 0].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
        axes[1, 0].set_title('City Summary Statistics')
        axes[1, 0].axis('off')
        
        # 5. Sample images (if available)
        if len(results) >= 3:
            sample_results = results[:3]
            for i, result in enumerate(sample_results):
                row, col = 1, i + 1
                if col < 3:
                    try:
                        image_path = result['image_path']
                        image = cv2.imread(image_path)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
                        # Resize for display
                        image = cv2.resize(image, (200, 200))
                        axes[row, col].imshow(image)
                        axes[row, col].set_title(f"Sample {i+1}: {result['total_trees']} trees")
                        axes[row, col].axis('off')
                    except:
                        axes[row, col].text(0.5, 0.5, f"Sample {i+1}\n(Image not available)", 
                                         ha='center', va='center')
                        axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'city_analysis_report.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save city statistics
        save_results(city_stats, os.path.join(output_dir, 'city_statistics.json'))

def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description='Tree Segmentation Inference')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth',
                        help='Path to trained model')
    parser.add_argument('--image_path', type=str, help='Path to single image')
    parser.add_argument('--image_dir', type=str, help='Path to image directory')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                        help='Output directory for results')
    parser.add_argument('--city_name', type=str, help='City name for analysis')
    parser.add_argument('--batch', action='store_true', help='Process batch of images')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = TreeSegmentationPredictor(args.model_path)
    
    if args.image_path:
        # Single image prediction
        results = predictor.predict_single_image(args.image_path, args.output_dir)
        print(f"\nResults for {args.image_path}:")
        print(f"Total trees: {results['total_trees']}")
        print(f"Tree coverage: {results['coverage_percentage']:.2f}%")
        
    elif args.image_dir and args.batch:
        # Batch prediction
        if args.city_name:
            # City-specific analysis
            city_stats = predictor.predict_city_analysis(args.image_dir, args.city_name, args.output_dir)
            if city_stats:
                print(f"\nCity Analysis Results for {args.city_name}:")
                print(f"Total images: {city_stats['total_images']}")
                print(f"Total trees: {city_stats['total_trees']}")
                print(f"Average coverage: {city_stats['average_coverage']:.2f}%")
        else:
            # Regular batch prediction
            results = predictor.predict_batch(args.image_dir, args.output_dir)
            
    else:
        print("Please provide either --image_path for single image or --image_dir with --batch for batch processing")
        print("Example usage:")
        print("  python predict.py --image_path path/to/image.jpg")
        print("  python predict.py --image_dir path/to/images --batch --output_dir results")
        print("  python predict.py --image_dir path/to/images --batch --city_name 'New York'")

if __name__ == "__main__":
    main()
