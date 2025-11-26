"""
Exploratory Data Analysis for Dental Fractured Instrument Detection
Master Thesis Project - 2025
Author: Computer Vision Expert
Target: Analyze panoramic X-ray images and annotations
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import cv2
from tqdm import tqdm
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)

class DentalDatasetEDA:
    """Comprehensive EDA for Dental X-ray Dataset"""
    
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.fractured_dir = self.root_dir / "Fractured"
        self.healthy_dir = self.root_dir / "Healthy"
        self.stats = defaultdict(dict)
        
    def get_image_files(self, directory):
        """Get all image files from directory"""
        return sorted(list(directory.glob("*.jpg")) + list(directory.glob("*.png")))
    
    def get_annotation_file(self, image_path):
        """Get corresponding annotation file for an image"""
        txt_path = image_path.with_suffix('.txt')
        return txt_path if txt_path.exists() else None
    
    def parse_annotation(self, txt_path):
        """Parse annotation file and extract line vectors (each 2 consecutive lines form a vector)"""
        if not txt_path or not txt_path.exists():
            return None
        
        try:
            with open(txt_path, 'r') as f:
                lines = f.readlines()
            
            # Filter out empty lines
            points = []
            for line in lines:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        x, y = float(parts[0]), float(parts[1])
                        points.append((x, y))
            
            # Each 2 consecutive points form a line/vector
            # vectors = [(p1, p2), (p3, p4), ...] where p1-p2 is one line, p3-p4 is another line
            vectors = []
            for i in range(0, len(points)-1, 2):
                start_point = points[i]
                end_point = points[i+1]
                vectors.append((start_point, end_point))
            
            return vectors if vectors else None
        except Exception as e:
            print(f"Error parsing {txt_path}: {e}")
            return None
    
    def analyze_images(self, image_paths, class_name):
        """Analyze image dimensions, quality, and statistics"""
        print(f"\n{'='*60}")
        print(f"Analyzing {class_name} images...")
        print(f"{'='*60}")
        
        widths, heights, aspect_ratios = [], [], []
        pixel_means, pixel_stds = [], []
        brightness_values, contrast_values = [], []
        file_sizes = []
        annotation_counts = []
        
        for img_path in tqdm(image_paths, desc=f"{class_name}"):
            # Load image with PIL (better Unicode path handling on Windows)
            try:
                from PIL import Image
                pil_img = Image.open(img_path).convert('L')  # Convert to grayscale
                img = np.array(pil_img)
            except Exception as e:
                print(f"\nWarning: Could not read {img_path.name}: {e}")
                continue
            
            # Dimensions
            h, w = img.shape
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h)
            
            # Pixel statistics
            pixel_means.append(img.mean())
            pixel_stds.append(img.std())
            
            # Brightness and contrast
            brightness_values.append(np.mean(img))
            contrast_values.append(np.std(img))
            
            # File size
            file_sizes.append(img_path.stat().st_size / 1024)  # KB
            
            # Annotations
            ann_path = self.get_annotation_file(img_path)
            coords = self.parse_annotation(ann_path)
            annotation_counts.append(len(coords) if coords else 0)
        
        # Store statistics
        self.stats[class_name] = {
            'count': len(image_paths),
            'widths': np.array(widths),
            'heights': np.array(heights),
            'aspect_ratios': np.array(aspect_ratios),
            'pixel_means': np.array(pixel_means),
            'pixel_stds': np.array(pixel_stds),
            'brightness': np.array(brightness_values),
            'contrast': np.array(contrast_values),
            'file_sizes': np.array(file_sizes),
            'annotation_counts': np.array(annotation_counts)
        }
        
        # Print summary
        print(f"\n{class_name} Dataset Summary:")
        print(f"  Total images: {len(image_paths)}")
        print(f"  Image dimensions:")
        print(f"    Width  - Mean: {np.mean(widths):.1f}, Std: {np.std(widths):.1f}, Range: [{np.min(widths)}, {np.max(widths)}]")
        print(f"    Height - Mean: {np.mean(heights):.1f}, Std: {np.std(heights):.1f}, Range: [{np.min(heights)}, {np.max(heights)}]")
        print(f"    Aspect Ratio - Mean: {np.mean(aspect_ratios):.3f}, Std: {np.std(aspect_ratios):.3f}")
        print(f"  Pixel intensity:")
        print(f"    Mean: {np.mean(pixel_means):.2f}, Std: {np.std(pixel_means):.2f}")
        print(f"  Brightness: {np.mean(brightness_values):.2f} ± {np.std(brightness_values):.2f}")
        print(f"  Contrast: {np.mean(contrast_values):.2f} ± {np.std(contrast_values):.2f}")
        print(f"  File size: {np.mean(file_sizes):.2f} KB ± {np.std(file_sizes):.2f}")
        print(f"  Annotations per image: {np.mean(annotation_counts):.2f} ± {np.std(annotation_counts):.2f}")
        
        return self.stats[class_name]
    
    def visualize_statistics(self, save_dir='outputs/eda'):
        """Create comprehensive visualizations"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Class distribution
        plt.figure(figsize=(10, 6))
        classes = list(self.stats.keys())
        counts = [self.stats[c]['count'] for c in classes]
        colors = ['#2ecc71', '#e74c3c']
        plt.bar(classes, counts, color=colors, alpha=0.7, edgecolor='black')
        plt.title('Class Distribution', fontsize=16, fontweight='bold')
        plt.ylabel('Number of Images', fontsize=12)
        plt.xlabel('Class', fontsize=12)
        for i, v in enumerate(counts):
            plt.text(i, v + 5, str(v), ha='center', fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Image dimensions comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for idx, metric in enumerate(['widths', 'heights', 'aspect_ratios', 'brightness']):
            ax = axes[idx // 2, idx % 2]
            for class_name in classes:
                data = self.stats[class_name][metric]
                ax.hist(data, bins=30, alpha=0.6, label=class_name, edgecolor='black')
            ax.set_title(f'{metric.replace("_", " ").title()} Distribution', fontweight='bold')
            ax.set_xlabel(metric.replace('_', ' ').title())
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/dimensions_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Pixel intensity analysis
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        for idx, metric in enumerate(['pixel_means', 'pixel_stds']):
            ax = axes[idx]
            for class_name in classes:
                data = self.stats[class_name][metric]
                ax.hist(data, bins=30, alpha=0.6, label=class_name, edgecolor='black')
            ax.set_title(f'{metric.replace("_", " ").title()} Distribution', fontweight='bold')
            ax.set_xlabel(metric.replace('_', ' ').title())
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/pixel_intensity.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Annotation statistics
        plt.figure(figsize=(12, 6))
        for class_name in classes:
            data = self.stats[class_name]['annotation_counts']
            plt.hist(data, bins=20, alpha=0.6, label=class_name, edgecolor='black')
        plt.title('Annotation Points per Image', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Annotation Points', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/annotation_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Box plots comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        metrics = ['widths', 'heights', 'brightness', 'contrast', 'file_sizes', 'annotation_counts']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            data_to_plot = [self.stats[c][metric] for c in classes]
            bp = ax.boxplot(data_to_plot, labels=classes, patch_artist=True)
            
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            
            ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/boxplot_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Visualizations saved to: {save_dir}/")
    
    def save_summary_report(self, save_path='outputs/eda/dataset_summary.json'):
        """Save comprehensive summary report"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        summary = {}
        for class_name, stats in self.stats.items():
            summary[class_name] = {
                'total_images': int(stats['count']),
                'dimensions': {
                    'width_mean': float(np.mean(stats['widths'])),
                    'width_std': float(np.std(stats['widths'])),
                    'height_mean': float(np.mean(stats['heights'])),
                    'height_std': float(np.std(stats['heights'])),
                    'aspect_ratio_mean': float(np.mean(stats['aspect_ratios'])),
                },
                'pixel_statistics': {
                    'mean_intensity': float(np.mean(stats['pixel_means'])),
                    'std_intensity': float(np.mean(stats['pixel_stds'])),
                    'brightness': float(np.mean(stats['brightness'])),
                    'contrast': float(np.mean(stats['contrast'])),
                },
                'annotations': {
                    'avg_points_per_image': float(np.mean(stats['annotation_counts'])),
                    'max_points': int(np.max(stats['annotation_counts'])),
                    'min_points': int(np.min(stats['annotation_counts'])),
                }
            }
        
        # Overall statistics
        total_images = sum(s['count'] for s in self.stats.values())
        summary['overall'] = {
            'total_images': total_images,
            'class_distribution': {c: s['count'] for c, s in self.stats.items()},
            'class_imbalance_ratio': float(self.stats['Fractured']['count'] / self.stats['Healthy']['count']),
            'recommended_class_weights': {
                'Healthy': float(self.stats['Fractured']['count'] / self.stats['Healthy']['count']),
                'Fractured': 1.0
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Summary report saved to: {save_path}")
        return summary
    
    def visualize_sample_images(self, n_samples=5, save_dir='outputs/eda'):
        """Visualize sample images with annotations"""
        os.makedirs(save_dir, exist_ok=True)
        
        for class_name in ['Fractured', 'Healthy']:
            fig, axes = plt.subplots(1, n_samples, figsize=(20, 4))
            directory = self.fractured_dir if class_name == 'Fractured' else self.healthy_dir
            image_files = self.get_image_files(directory)
            
            # Random sample
            samples = np.random.choice(image_files, min(n_samples, len(image_files)), replace=False)
            
            for idx, img_path in enumerate(samples):
                ax = axes[idx] if n_samples > 1 else axes
                
                # Load image with PIL (better Unicode path handling)
                try:
                    from PIL import Image
                    pil_img = Image.open(img_path).convert('L')
                    img = np.array(pil_img)
                except Exception as e:
                    print(f"\nWarning: Could not read {img_path.name}: {e}")
                    continue
                
                # Get annotations
                ann_path = self.get_annotation_file(img_path)
                coords = self.parse_annotation(ann_path)
                
                # Display
                ax.imshow(img, cmap='gray')
                
                # Plot annotations as line vectors if available
                if coords:
                    # coords is a list of tuples: [(start_point, end_point), ...]
                    # Each tuple represents one line/vector
                    for vector in coords:
                        start_point, end_point = vector
                        # Draw line from start to end
                        ax.plot([start_point[0], end_point[0]], 
                               [start_point[1], end_point[1]], 
                               'r-', linewidth=2, alpha=0.7)
                        # Mark start and end points
                        ax.scatter([start_point[0], end_point[0]], 
                                 [start_point[1], end_point[1]], 
                                 c='red', s=40, marker='o', edgecolors='yellow', linewidths=1)
                
                ax.set_title(f'{img_path.stem}\n{img.shape[1]}x{img.shape[0]} - {len(coords) if coords else 0} lines', fontsize=10)
                ax.axis('off')
            
            plt.suptitle(f'{class_name} - Sample Images with Annotations', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{save_dir}/samples_{class_name.lower()}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✅ Sample images saved to: {save_dir}/")
    
    def run_complete_eda(self):
        """Run complete EDA pipeline"""
        print("\n" + "="*80)
        print(" DENTAL FRACTURED INSTRUMENT DETECTION - EXPLORATORY DATA ANALYSIS")
        print("="*80)
        
        # Analyze both classes
        fractured_images = self.get_image_files(self.fractured_dir)
        healthy_images = self.get_image_files(self.healthy_dir)
        
        self.analyze_images(fractured_images, 'Fractured')
        self.analyze_images(healthy_images, 'Healthy')
        
        # Visualizations
        print("\n" + "="*60)
        print("Creating visualizations...")
        print("="*60)
        self.visualize_statistics()
        self.visualize_sample_images()
        
        # Summary report
        summary = self.save_summary_report()
        
        print("\n" + "="*80)
        print(" EDA COMPLETE!")
        print("="*80)
        print(f"\nDataset Summary:")
        print(f"  Total Images: {summary['overall']['total_images']}")
        print(f"  Fractured: {summary['Fractured']['total_images']}")
        print(f"  Healthy: {summary['Healthy']['total_images']}")
        print(f"  Class Imbalance Ratio: {summary['overall']['class_imbalance_ratio']:.2f}:1")
        print(f"\nRecommended Class Weights:")
        print(f"  Healthy: {summary['overall']['recommended_class_weights']['Healthy']:.2f}")
        print(f"  Fractured: {summary['overall']['recommended_class_weights']['Fractured']:.2f}")
        
        return summary


if __name__ == "__main__":
    # Set paths
    ROOT_DIR = r"c:\Users\maspe\OneDrive\Masaüstü\masterthesis\Dataset_2021\Dataset_2021\Dataset"
    
    # Run EDA
    eda = DentalDatasetEDA(ROOT_DIR)
    summary = eda.run_complete_eda()
