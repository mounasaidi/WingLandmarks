"""
Tsetse Wing Landmark Dataset Preparation
Prepares annotated landmark data for deep learning training
Splits dataset into training and testing sets
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm
import argparse

class TsetseWingDatasetPreparator:
    """Prepare tsetse wing landmark dataset for deep learning"""
    
    def __init__(self, annotations_dir, output_dir, test_size=0.2, val_size=0.1, 
                 img_size=(512, 512), heatmap_sigma=3, random_seed=42):
        """
        Args:
            annotations_dir: Directory containing processed annotations from segmentation code
            output_dir: Output directory for prepared dataset
            test_size: Fraction of data for testing (default: 0.2)
            val_size: Fraction of training data for validation (default: 0.1)
            img_size: Target image size (width, height)
            heatmap_sigma: Sigma for Gaussian heatmap generation
            random_seed: Random seed for reproducibility
        """
        self.annotations_dir = Path(annotations_dir)
        self.output_dir = Path(output_dir)
        self.test_size = test_size
        self.val_size = val_size
        self.img_size = img_size
        self.heatmap_sigma = heatmap_sigma
        self.random_seed = random_seed
        
        # Create output directories
        self.splits = ['train', 'val', 'test']
        for split in self.splits:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'heatmaps').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'annotations').mkdir(parents=True, exist_ok=True)
    
    def generate_gaussian_heatmap(self, shape, landmarks, sigma):
        """Generate Gaussian heatmap for landmarks"""
        heatmap = np.zeros(shape, dtype=np.float32)
        
        for x, y in landmarks:
            # Create coordinate grids
            y_grid, x_grid = np.ogrid[0:shape[0], 0:shape[1]]
            
            # Gaussian formula
            gaussian = np.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * sigma**2))
            
            # Add to heatmap (take maximum to handle overlapping landmarks)
            heatmap = np.maximum(heatmap, gaussian)
        
        return heatmap
    
    def load_annotations(self, image_folder):
        """Load annotations from a processed image folder"""
        annotation_file = image_folder / 'annotations.json'
        
        if not annotation_file.exists():
            return None
        
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        # Extract landmarks
        landmarks = [(pt['x'], pt['y']) for pt in data['vein_junctions']]
        
        return {
            'landmarks': landmarks,
            'image_name': data['image_name'],
            'original_size': (data['image_size']['width'], data['image_size']['height'])
        }
    
    def preprocess_image_and_landmarks(self, image_path, landmarks, original_size):
        """Resize image and adjust landmark coordinates"""
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            return None, None
        
        # Get original dimensions
        orig_h, orig_w = img.shape[:2]
        target_w, target_h = self.img_size
        
        # Resize image
        img_resized = cv2.resize(img, (target_w, target_h))
        
        # Scale landmarks
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        
        landmarks_scaled = [(int(x * scale_x), int(y * scale_y)) for x, y in landmarks]
        
        return img_resized, landmarks_scaled
    
    def prepare_dataset(self):
        """Prepare complete dataset with train/val/test splits"""
        
        print("🔍 Scanning for annotated images...")
        
        # Find all image folders with annotations
        image_folders = []
        for folder in self.annotations_dir.iterdir():
            if folder.is_dir() and (folder / 'annotations.json').exists():
                image_folders.append(folder)
        
        if not image_folders:
            print("❌ No annotated images found!")
            return
        
        print(f"✅ Found {len(image_folders)} annotated images")
        
        # Load all annotations
        dataset_samples = []
        for folder in tqdm(image_folders, desc="Loading annotations"):
            annotations = self.load_annotations(folder)
            if annotations and len(annotations['landmarks']) > 0:
                original_img_path = folder / 'original_image.jpg'
                if original_img_path.exists():
                    dataset_samples.append({
                        'folder': folder,
                        'image_path': original_img_path,
                        'annotations': annotations
                    })
        
        print(f"📊 Valid samples: {len(dataset_samples)}")
        
        if len(dataset_samples) == 0:
            print("❌ No valid samples found!")
            return
        
        # Split dataset
        print("\n✂️ Splitting dataset...")
        
        # First split: train+val vs test
        train_val_samples, test_samples = train_test_split(
            dataset_samples, 
            test_size=self.test_size, 
            random_state=self.random_seed
        )
        
        # Second split: train vs val
        train_samples, val_samples = train_test_split(
            train_val_samples,
            test_size=self.val_size / (1 - self.test_size),
            random_state=self.random_seed
        )
        
        splits_data = {
            'train': train_samples,
            'val': val_samples,
            'test': test_samples
        }
        
        print(f"   • Train: {len(train_samples)} samples")
        print(f"   • Validation: {len(val_samples)} samples")
        print(f"   • Test: {len(test_samples)} samples")
        
        # Process each split
        dataset_info = {
            'image_size': self.img_size,
            'heatmap_sigma': self.heatmap_sigma,
            'num_classes': 1,  # Single class: vein junction
            'splits': {}
        }
        
        for split_name, samples in splits_data.items():
            print(f"\n📦 Processing {split_name} set...")
            
            split_info = []
            
            for idx, sample in enumerate(tqdm(samples, desc=f"Processing {split_name}")):
                img_name = f"{split_name}_{idx:04d}"
                
                # Preprocess image and landmarks
                img_resized, landmarks_scaled = self.preprocess_image_and_landmarks(
                    sample['image_path'],
                    sample['annotations']['landmarks'],
                    sample['annotations']['original_size']
                )
                
                if img_resized is None:
                    continue
                
                # Generate heatmap
                heatmap = self.generate_gaussian_heatmap(
                    self.img_size[::-1],  # (height, width)
                    landmarks_scaled,
                    self.heatmap_sigma
                )
                
                # Save image
                img_out_path = self.output_dir / split_name / 'images' / f"{img_name}.jpg"
                cv2.imwrite(str(img_out_path), img_resized)
                
                # Save heatmap
                heatmap_out_path = self.output_dir / split_name / 'heatmaps' / f"{img_name}.npy"
                np.save(str(heatmap_out_path), heatmap)
                
                # Save heatmap visualization
                heatmap_vis = (heatmap * 255).astype(np.uint8)
                heatmap_colored = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
                cv2.imwrite(str(self.output_dir / split_name / 'heatmaps' / f"{img_name}_vis.jpg"), 
                           heatmap_colored)
                
                # Save annotations (JSON)
                annotation_data = {
                    'image_name': img_name,
                    'original_image': str(sample['annotations']['image_name']),
                    'landmarks': [{'x': int(x), 'y': int(y)} for x, y in landmarks_scaled],
                    'num_landmarks': len(landmarks_scaled),
                    'image_size': self.img_size
                }
                
                annotation_out_path = self.output_dir / split_name / 'annotations' / f"{img_name}.json"
                with open(annotation_out_path, 'w') as f:
                    json.dump(annotation_data, f, indent=2)
                
                split_info.append(annotation_data)
            
            dataset_info['splits'][split_name] = {
                'num_samples': len(split_info),
                'samples': split_info
            }
        
        # Save dataset metadata
        metadata_path = self.output_dir / 'dataset_info.json'
        with open(metadata_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        # Create summary report
        self._create_summary_report(dataset_info)
        
        print("\n" + "="*60)
        print("✅ DATASET PREPARATION COMPLETE!")
        print("="*60)
        print(f"📁 Dataset saved to: {self.output_dir}")
        print(f"\n📊 Dataset Summary:")
        print(f"   • Train samples: {dataset_info['splits']['train']['num_samples']}")
        print(f"   • Validation samples: {dataset_info['splits']['val']['num_samples']}")
        print(f"   • Test samples: {dataset_info['splits']['test']['num_samples']}")
        print(f"   • Total samples: {sum(s['num_samples'] for s in dataset_info['splits'].values())}")
        print(f"   • Image size: {self.img_size}")
        print(f"   • Heatmap sigma: {self.heatmap_sigma}")
        print("="*60)
    
    def _create_summary_report(self, dataset_info):
        """Create summary report"""
        report_path = self.output_dir / 'dataset_summary.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("TSETSE WING LANDMARK DETECTION DATASET\n")
            f.write("="*60 + "\n\n")
            
            f.write("DATASET CONFIGURATION:\n")
            f.write("-"*60 + "\n")
            f.write(f"Image Size: {dataset_info['image_size']}\n")
            f.write(f"Heatmap Sigma: {dataset_info['heatmap_sigma']}\n")
            f.write(f"Number of Classes: {dataset_info['num_classes']}\n\n")
            
            f.write("DATASET SPLITS:\n")
            f.write("-"*60 + "\n")
            
            for split_name, split_data in dataset_info['splits'].items():
                f.write(f"\n{split_name.upper()} SET:\n")
                f.write(f"  Samples: {split_data['num_samples']}\n")
                
                if split_data['samples']:
                    num_landmarks = [s['num_landmarks'] for s in split_data['samples']]
                    f.write(f"  Avg landmarks per image: {np.mean(num_landmarks):.1f}\n")
                    f.write(f"  Min landmarks: {np.min(num_landmarks)}\n")
                    f.write(f"  Max landmarks: {np.max(num_landmarks)}\n")
                    f.write(f"  Total landmarks: {np.sum(num_landmarks)}\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("DIRECTORY STRUCTURE:\n")
            f.write("-"*60 + "\n")
            f.write("dataset/\n")
            f.write("  ├── train/\n")
            f.write("  │   ├── images/       # Training images\n")
            f.write("  │   ├── heatmaps/     # Training heatmaps (.npy and _vis.jpg)\n")
            f.write("  │   └── annotations/  # Training annotations (.json)\n")
            f.write("  ├── val/\n")
            f.write("  │   ├── images/       # Validation images\n")
            f.write("  │   ├── heatmaps/     # Validation heatmaps\n")
            f.write("  │   └── annotations/  # Validation annotations\n")
            f.write("  ├── test/\n")
            f.write("  │   ├── images/       # Test images\n")
            f.write("  │   ├── heatmaps/     # Test heatmaps\n")
            f.write("  │   └── annotations/  # Test annotations\n")
            f.write("  ├── dataset_info.json    # Dataset metadata\n")
            f.write("  └── dataset_summary.txt  # This file\n")


def main():
    parser = argparse.ArgumentParser(description='Prepare tsetse wing landmark dataset')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input directory with annotations (output from segmentation code)')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output directory for prepared dataset')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set fraction (default: 0.2)')
    parser.add_argument('--val-size', type=float, default=0.1,
                       help='Validation set fraction of training data (default: 0.1)')
    parser.add_argument('--img-size', type=int, nargs=2, default=[512, 512],
                       help='Target image size [width height] (default: 512 512)')
    parser.add_argument('--sigma', type=float, default=3.0,
                       help='Heatmap Gaussian sigma (default: 3.0)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    print("🦋 TSETSE WING LANDMARK DATASET PREPARATION")
    print("="*60)
    
    preparator = TsetseWingDatasetPreparator(
        annotations_dir=args.input,
        output_dir=args.output,
        test_size=args.test_size,
        val_size=args.val_size,
        img_size=tuple(args.img_size),
        heatmap_sigma=args.sigma,
        random_seed=args.seed
    )
    
    preparator.prepare_dataset()


if __name__ == "__main__":
    main()