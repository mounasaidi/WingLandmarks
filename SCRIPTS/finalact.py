"""
Local GPU-Accelerated Wing Vein Intersection Detection
Processes folder of butterfly wing images and creates dataset with vein junction annotations
Uses GPU acceleration when available (PyTorch + OpenCV)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, remove_small_objects
from skimage.filters import frangi
import scipy.ndimage as ndi
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import json
import argparse
import shutil
import torch
import gc

# ==================== GPU CONFIGURATION ====================
def setup_gpu():
    """Check and configure GPU usage with proper detection"""
    gpu_available = False
    gpu_info = "CPU mode"
    
    # Check PyTorch CUDA
    if torch.cuda.is_available():
        gpu_available = True
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_info = f"PyTorch CUDA: {device_count} device(s), {device_name}, {memory_total:.1f}GB"
        print(f"✅ GPU Available - {gpu_info}")
        
        # Set device
        torch.cuda.set_device(0)
        
        # Optimize CUDA settings
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        return True, 0
    
    # Check OpenCV CUDA (secondary check)
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            gpu_available = True
            device_count = cv2.cuda.getCudaEnabledDeviceCount()
            gpu_info = f"OpenCV CUDA: {device_count} device(s)"
            print(f"✅ GPU Available (OpenCV) - {gpu_info}")
            return True, 0
    except:
        pass
    
    if not gpu_available:
        print("⚠️ No GPU detected - Running on CPU")
        print("   For GPU acceleration, ensure CUDA is installed:")
        print("   • PyTorch with CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("   • OpenCV with CUDA: Build from source or use cuda-enabled builds")
    
    return gpu_available, -1

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

def get_memory_info():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"GPU: {allocated:.1f}GB used, {reserved:.1f}GB reserved, {total:.1f}GB total"
    return "CPU mode"

# ==================== GPU-ACCELERATED IMAGE PROCESSING ====================
class GPUImageProcessor:
    """GPU-accelerated image processing operations"""
    
    def __init__(self, use_gpu=False, device=0):
        self.use_gpu = use_gpu
        self.device = device
        
        if self.use_gpu:
            self.torch_device = torch.device(f'cuda:{device}')
            print(f"🚀 GPU Processor initialized on device {device}")
        else:
            self.torch_device = torch.device('cpu')
            print("💻 CPU Processor initialized")
    
    def to_tensor(self, img):
        """Convert numpy array to PyTorch tensor on GPU"""
        if self.use_gpu:
            tensor = torch.from_numpy(img).to(self.torch_device)
            return tensor
        return img
    
    def to_numpy(self, tensor):
        """Convert PyTorch tensor back to numpy"""
        if self.use_gpu and isinstance(tensor, torch.Tensor):
            return tensor.cpu().numpy()
        return tensor
    
    def gaussian_blur_gpu(self, img, kernel_size=(9, 9), sigma=2):
        """GPU-accelerated Gaussian blur using PyTorch"""
        if not self.use_gpu:
            return cv2.GaussianBlur(img, kernel_size, sigma)
        
        # Convert to tensor
        img_tensor = self.to_tensor(img.astype(np.float32))
        
        # Create Gaussian kernel
        kernel_1d = self._gaussian_kernel_1d(kernel_size[0], sigma)
        kernel_1d = kernel_1d.to(self.torch_device)
        
        # Apply separable convolution (faster than 2D)
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        
        # Horizontal blur
        img_tensor = torch.nn.functional.conv2d(
            img_tensor, 
            kernel_1d.view(1, 1, 1, -1),
            padding=(0, kernel_size[0]//2)
        )
        
        # Vertical blur
        img_tensor = torch.nn.functional.conv2d(
            img_tensor,
            kernel_1d.view(1, 1, -1, 1),
            padding=(kernel_size[0]//2, 0)
        )
        
        result = img_tensor.squeeze().clamp(0, 255)
        return self.to_numpy(result).astype(np.uint8)
    
    def _gaussian_kernel_1d(self, size, sigma):
        """Generate 1D Gaussian kernel"""
        coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
        kernel = torch.exp(-(coords**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel
    
    def morphology_gpu(self, img, operation, kernel_size=5, iterations=1):
        """GPU-accelerated morphological operations"""
        if not self.use_gpu:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            if operation == 'close':
                return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iterations)
            elif operation == 'open':
                return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)
            elif operation == 'dilate':
                return cv2.dilate(img, kernel, iterations=iterations)
            elif operation == 'erode':
                return cv2.erode(img, kernel, iterations=iterations)
        
        # Use PyTorch for GPU morphology
        img_tensor = self.to_tensor(img.astype(np.float32) / 255.0)
        kernel = torch.ones(1, 1, kernel_size, kernel_size).to(self.torch_device)
        
        for _ in range(iterations):
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
            
            if operation in ['dilate', 'close']:
                img_tensor = torch.nn.functional.max_pool2d(
                    img_tensor,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size//2
                )
            elif operation in ['erode', 'open']:
                img_tensor = -torch.nn.functional.max_pool2d(
                    -img_tensor,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size//2
                )
            
            img_tensor = img_tensor.squeeze()
        
        result = (img_tensor * 255).clamp(0, 255)
        return self.to_numpy(result).astype(np.uint8)
    
    def threshold_gpu(self, img):
        """GPU-accelerated Otsu thresholding"""
        if not self.use_gpu:
            _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return thresh
        
        # Use CPU for Otsu (complex histogram calculation)
        # But can do simple thresholding on GPU
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

# ==================== CORE PROCESSING FUNCTIONS ====================
def clean_and_thin_skeleton(skel_img, close_kernel_size=3, prune_iters=8):
    """Clean and thin skeleton image"""
    sk = (skel_img > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel_size, close_kernel_size))
    closed = cv2.morphologyEx(sk * 255, cv2.MORPH_CLOSE, kernel, iterations=1)
    thin = skeletonize(closed // 255).astype(np.uint8)

    neigh_kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    pruned = thin.copy()
    for _ in range(prune_iters):
        neighbor_count = cv2.filter2D(pruned, -1, neigh_kernel)
        endpoints = (pruned == 1) & (neighbor_count == 1)
        if not endpoints.any():
            break
        pruned[endpoints] = 0
    return pruned

def find_vein_junctions(skeleton_img, min_degree=3):
    """Find vein-vein junctions (RED MARKS ONLY)"""
    sk_clean = clean_and_thin_skeleton(skeleton_img, close_kernel_size=3, prune_iters=6)
    kernel = np.ones((3,3), dtype=np.uint8)
    kernel[1,1] = 0
    neighbor_count = cv2.filter2D(sk_clean, -1, kernel)
    junctions_mask = (sk_clean == 1) & (neighbor_count >= min_degree)
    junctions_label, _ = ndi.label(junctions_mask)
    ys, xs = np.nonzero(junctions_label)
    points = list(zip(xs, ys))
    return points, sk_clean

def cluster_points_by_components(points, image_shape, cluster_radius=12):
    """Cluster nearby points"""
    if len(points) == 0:
        return []
    canvas = np.zeros(image_shape, dtype=np.uint8)
    for (x, y) in points:
        cv2.circle(canvas, (int(x), int(y)), cluster_radius, 255, -1)
    num_labels, labels = cv2.connectedComponents(canvas)
    clustered = []
    for lab in range(1, num_labels):
        ys, xs = np.nonzero(labels == lab)
        if len(xs) == 0:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())
        clustered.append((cx, cy))
    return clustered

def process_single_image(image_path, output_dir, gpu_processor, save_visualizations=True):
    """Process a single butterfly wing image - RED MARKS ONLY with GPU acceleration"""

    # Create output subdirectory for this image
    img_name = Path(image_path).stem
    img_output_dir = os.path.join(output_dir, img_name)
    os.makedirs(img_output_dir, exist_ok=True)

    # Load image
    img_color = cv2.imread(str(image_path))
    if img_color is None:
        return None, f"Could not load image: {image_path}"

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape

    # Preprocessing with GPU acceleration
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(img_gray)

    # GPU-accelerated thresholding
    thresh = gpu_processor.threshold_gpu(enhanced)
    
    if np.mean(img_gray[thresh == 255]) > np.mean(img_gray[thresh == 0]):
        thresh = cv2.bitwise_not(thresh)

    # GPU-accelerated morphology
    thresh = gpu_processor.morphology_gpu(thresh, 'close', kernel_size=5, iterations=2)
    thresh = gpu_processor.morphology_gpu(thresh, 'open', kernel_size=5, iterations=2)

    # Find wing contour
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    wing_mask = np.zeros_like(img_gray)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(wing_mask, [largest], -1, 255, thickness=cv2.FILLED)
        selected_contour = largest
    else:
        wing_mask[:] = 255
        selected_contour = None

    masked = cv2.bitwise_and(enhanced, wing_mask)

    # Vein detection - Thick Morphological (GPU-accelerated)
    inverted = cv2.bitwise_not(masked)
    blurred_heavy = gpu_processor.gaussian_blur_gpu(inverted, (9, 9), 2)
    thick_binary = gpu_processor.threshold_gpu(blurred_heavy)
    
    thick_binary_labeled = thick_binary.astype(bool)
    thick_cleaned = remove_small_objects(thick_binary_labeled, min_size=500)
    thick_cleaned = (thick_cleaned * 255).astype(np.uint8)
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    thick_closed = cv2.morphologyEx(thick_cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    thick_closed = cv2.bitwise_and(thick_closed, wing_mask)
    thick_skeleton = skeletonize(thick_closed // 255)
    thick_skeleton = (thick_skeleton * 255).astype(np.uint8)
    thick_skeleton = cv2.bitwise_and(thick_skeleton, wing_mask)

    # Vein detection - Frangi (CPU - complex filter)
    masked_normalized = masked.astype(np.float64) / 255.0
    frangi_thick = frangi(masked_normalized, sigmas=range(3, 10), black_ridges=True)
    frangi_thick_norm = (frangi_thick * 255).astype(np.uint8)
    _, frangi_binary = cv2.threshold(frangi_thick_norm, 50, 255, cv2.THRESH_BINARY)
    frangi_binary = cv2.bitwise_and(frangi_binary, wing_mask)
    frangi_labeled = frangi_binary.astype(bool)
    frangi_cleaned = remove_small_objects(frangi_labeled, min_size=300)
    frangi_cleaned = (frangi_cleaned * 255).astype(np.uint8)
    frangi_cleaned = cv2.bitwise_and(frangi_cleaned, wing_mask)
    frangi_skeleton = skeletonize(frangi_cleaned // 255)
    frangi_skeleton = (frangi_skeleton * 255).astype(np.uint8)
    frangi_skeleton = cv2.bitwise_and(frangi_skeleton, wing_mask)

    # Combined vein skeleton
    vein_skeleton = cv2.bitwise_or(thick_skeleton, frangi_skeleton)
    vein_skeleton = cv2.bitwise_and(vein_skeleton, wing_mask)

    # Detect ONLY vein junctions (RED MARKS)
    vein_junctions_raw, _ = find_vein_junctions(vein_skeleton, min_degree=3)
    vein_junctions_clustered = cluster_points_by_components(vein_junctions_raw, vein_skeleton.shape, cluster_radius=8)

    # Save results
    results = {
        'image_name': img_name,
        'image_path': str(image_path),
        'width': w,
        'height': h,
        'vein_junctions': vein_junctions_clustered,
        'num_vein_junctions': len(vein_junctions_clustered)
    }

    # Save visualizations with RED MARKS ONLY
    if save_visualizations:
        # Annotated image with RED marks
        annotated_img = img_color.copy()
        vein_overlay = cv2.cvtColor(vein_skeleton, cv2.COLOR_GRAY2BGR)
        vein_overlay[vein_skeleton > 0] = [200, 200, 200]
        annotated_img = cv2.addWeighted(annotated_img, 1.0, vein_overlay, 0.4, 0)

        # Draw RED marks for vein junctions
        for x, y in vein_junctions_clustered:
            cv2.circle(annotated_img, (x, y), 8, (0, 0, 255), -1)  # Red fill
            cv2.circle(annotated_img, (x, y), 10, (0, 255, 255), 2)  # Yellow outline

        # Draw contour
        if selected_contour is not None:
            cv2.drawContours(annotated_img, [selected_contour], -1, (0, 255, 0), 2)

        # Save images
        cv2.imwrite(os.path.join(img_output_dir, "annotated_image.jpg"), annotated_img)
        cv2.imwrite(os.path.join(img_output_dir, "vein_skeleton.jpg"), vein_skeleton)
        
        # Copy original image
        shutil.copy(image_path, os.path.join(img_output_dir, "original_image.jpg"))

    # Save coordinates in multiple formats
    # CSV format
    with open(os.path.join(img_output_dir, "vein_junctions.csv"), 'w') as f:
        f.write("ID,X,Y,Type\n")
        for idx, (x, y) in enumerate(vein_junctions_clustered, 1):
            f.write(f"{idx},{x},{y},vein_junction\n")

    # JSON format (for easy loading)
    json_data = {
        'image_name': img_name,
        'image_size': {'width': w, 'height': h},
        'vein_junctions': [{'id': i+1, 'x': int(x), 'y': int(y)} for i, (x, y) in enumerate(vein_junctions_clustered)]
    }
    with open(os.path.join(img_output_dir, "annotations.json"), 'w') as f:
        json.dump(json_data, f, indent=2)

    # YOLO format (normalized coordinates)
    with open(os.path.join(img_output_dir, "yolo_annotations.txt"), 'w') as f:
        for x, y in vein_junctions_clustered:
            # YOLO format: class_id x_center y_center width height (normalized)
            x_norm = x / w
            y_norm = y / h
            # Small box around point (e.g., 16x16 pixels normalized)
            box_w = 16 / w
            box_h = 16 / h
            f.write(f"0 {x_norm:.6f} {y_norm:.6f} {box_w:.6f} {box_h:.6f}\n")

    return results, None

# ==================== BATCH PROCESSING ====================
def process_dataset(input_folder, output_base_dir="./results", save_visualizations=True, use_gpu=True):
    """Process entire dataset of images from a folder with GPU acceleration"""

    # Setup GPU
    gpu_available, device = setup_gpu()
    if not gpu_available:
        use_gpu = False
    
    gpu_processor = GPUImageProcessor(use_gpu=use_gpu, device=device)

    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_paths = []
    
    input_path = Path(input_folder)
    if not input_path.exists():
        print(f"❌ Input folder not found: {input_folder}")
        return None, None, None, None

    for ext in image_extensions:
        image_paths.extend(list(input_path.glob(f'*{ext}')))
        image_paths.extend(list(input_path.glob(f'*{ext.upper()}')))

    if not image_paths:
        print(f"❌ No images found in: {input_folder}")
        return None, None, None, None

    print(f"✅ Found {len(image_paths)} image(s) in {input_folder}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_base_dir, f"batch_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n🦋 Processing {len(image_paths)} images...")
    print(f"💻 Mode: {'GPU' if use_gpu else 'CPU'}")
    print(f"📊 {get_memory_info()}")
    print("="*60)

    all_results = []
    errors = []

    for img_path in tqdm(image_paths, desc="Processing images"):
        result, error = process_single_image(img_path, output_dir, gpu_processor, save_visualizations)
        if error:
            errors.append({'image': str(img_path), 'error': error})
        else:
            all_results.append(result)
        
        # Clear GPU memory periodically
        if use_gpu and len(all_results) % 10 == 0:
            clear_gpu_memory()

    # Create summary DataFrame
    summary_data = []
    for r in all_results:
        summary_data.append({
            'Image_Name': r['image_name'],
            'Width': r['width'],
            'Height': r['height'],
            'Vein_Junctions': r['num_vein_junctions']
        })

    df_summary = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, "dataset_summary.csv")
    df_summary.to_csv(summary_path, index=False)

    # Create master annotations file (all images)
    master_annotations = []
    for r in all_results:
        for i, (x, y) in enumerate(r['vein_junctions'], 1):
            master_annotations.append({
                'image_name': r['image_name'],
                'junction_id': i,
                'x': x,
                'y': y,
                'image_width': r['width'],
                'image_height': r['height']
            })
    
    df_annotations = pd.DataFrame(master_annotations)
    annotations_path = os.path.join(output_dir, "all_annotations.csv")
    df_annotations.to_csv(annotations_path, index=False)

    # Create COCO format annotations (for object detection)
    coco_format = {
        'info': {
            'description': 'Butterfly Wing Vein Junctions Dataset',
            'date_created': datetime.now().isoformat(),
            'processing_mode': 'GPU' if use_gpu else 'CPU'
        },
        'categories': [
            {'id': 0, 'name': 'vein_junction', 'supercategory': 'landmark'}
        ],
        'images': [],
        'annotations': []
    }

    annotation_id = 1
    for img_id, r in enumerate(all_results, 1):
        coco_format['images'].append({
            'id': img_id,
            'file_name': f"{r['image_name']}/original_image.jpg",
            'width': r['width'],
            'height': r['height']
        })

        for x, y in r['vein_junctions']:
            coco_format['annotations'].append({
                'id': annotation_id,
                'image_id': img_id,
                'category_id': 0,
                'bbox': [x-8, y-8, 16, 16],  # Small box around point
                'area': 256,
                'iscrowd': 0,
                'keypoints': [x, y, 2],  # x, y, visibility
                'num_keypoints': 1
            })
            annotation_id += 1

    with open(os.path.join(output_dir, "coco_annotations.json"), 'w') as f:
        json.dump(coco_format, f, indent=2)

    # Save detailed report
    report_path = os.path.join(output_dir, "processing_report.txt")
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("BUTTERFLY WING VEIN JUNCTION DETECTION (RED MARKS ONLY)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Processing Mode: {'GPU' if use_gpu else 'CPU'}\n")
        f.write(f"GPU Info: {get_memory_info()}\n")
        f.write(f"Input Folder: {input_folder}\n")
        f.write(f"Output Folder: {output_dir}\n")
        f.write(f"Total Images: {len(image_paths)}\n")
        f.write(f"Successfully Processed: {len(all_results)}\n")
        f.write(f"Errors: {len(errors)}\n\n")

        if all_results:
            f.write("SUMMARY STATISTICS:\n")
            f.write("-"*60 + "\n")
            f.write(f"Average Vein Junctions per Image: {df_summary['Vein_Junctions'].mean():.1f}\n")
            f.write(f"Min Vein Junctions: {df_summary['Vein_Junctions'].min()}\n")
            f.write(f"Max Vein Junctions: {df_summary['Vein_Junctions'].max()}\n")
            f.write(f"Total Vein Junctions: {df_summary['Vein_Junctions'].sum()}\n\n")

        f.write("\nOUTPUT FILES:\n")
        f.write("-"*60 + "\n")
        f.write("• dataset_summary.csv - Summary statistics per image\n")
        f.write("• all_annotations.csv - All junction coordinates\n")
        f.write("• coco_annotations.json - COCO format annotations\n")
        f.write("• processing_report.txt - This report\n")
        f.write("\nPer-image folders contain:\n")
        f.write("  - original_image.jpg - Original image\n")
        f.write("  - annotated_image.jpg - Image with RED marks\n")
        f.write("  - vein_skeleton.jpg - Vein skeleton\n")
        f.write("  - vein_junctions.csv - Junction coordinates (CSV)\n")
        f.write("  - annotations.json - Junction coordinates (JSON)\n")
        f.write("  - yolo_annotations.txt - YOLO format annotations\n")

        if errors:
            f.write("\n\nERRORS:\n")
            f.write("-"*60 + "\n")
            for err in errors:
                f.write(f"• {err['image']}: {err['error']}\n")

    print("\n" + "="*60)
    print("✅ BATCH PROCESSING COMPLETE!")
    print("="*60)
    print(f"📁 Results saved in: {output_dir}")
    print(f"\n📊 Summary:")
    print(f"   • Successfully processed: {len(all_results)}/{len(image_paths)} images")
    print(f"   • Errors: {len(errors)}")
    if all_results:
        print(f"   • Total vein junctions detected: {df_summary['Vein_Junctions'].sum()}")
        print(f"   • Average junctions per image: {df_summary['Vein_Junctions'].mean():.1f}")
    print(f"\n📄 Dataset files created:")
    print(f"   • {annotations_path}")
    print(f"   • {os.path.join(output_dir, 'coco_annotations.json')}")
    print(f"   • {summary_path}")
    print(f"\n💻 Processing mode: {'GPU' if use_gpu else 'CPU'}")
    print(f"📊 {get_memory_info()}")
    print("="*60)

    # Final GPU cleanup
    if use_gpu:
        clear_gpu_memory()

    return output_dir, df_summary, all_results, errors

# ==================== MAIN EXECUTION ====================
def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Process butterfly wing images for vein junction detection')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input folder containing images')
    parser.add_argument('--output', '-o', type=str, default='./results',
                        help='Output folder for results (default: ./results)')
    parser.add_argument('--no-viz', action='store_true',
                        help='Skip saving visualization images')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU processing (disable GPU)')

    args = parser.parse_args()

    print("🦋 BUTTERFLY WING VEIN JUNCTION DETECTION")
    print("="*60)
    
    # Check GPU availability
    print("\n🔍 Checking GPU availability...")
    gpu_available, _ = setup_gpu()
    
    use_gpu = gpu_available and not args.cpu
    
    if args.cpu:
        print("⚙️ CPU mode forced by user")
        use_gpu = False
    
    # Process dataset
    output_dir, df_summary, all_results, errors = process_dataset(
        args.input,
        output_base_dir=args.output,
        save_visualizations=not args.no_viz,
        use_gpu=use_gpu
    )

    if output_dir:
        print("\n✅ Processing complete!")
        print(f"📂 Check results in: {output_dir}")
    else:
        print("\n❌ Processing failed!")

if __name__ == "__main__":
    main()