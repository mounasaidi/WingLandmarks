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
    gpu_available = False
    if torch.cuda.is_available():
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        gpu_available = True
        return True, 0

    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            print("✅ GPU Available (OpenCV CUDA)")
            return True, 0
    except:
        pass

    print("⚠️ No GPU detected — running CPU mode")
    return False, -1

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

def get_memory_info():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"GPU: {allocated:.1f}GB used / {total:.1f}GB total"
    return "CPU mode"

# ==================== GPU-ACCELERATED IMAGE PROCESSOR ====================
class GPUImageProcessor:
    def __init__(self, use_gpu=False, device=0):
        self.use_gpu = use_gpu
        self.device = device
        self.torch_device = torch.device(f'cuda:{device}' if use_gpu else 'cpu')
        print("🚀 GPU Mode" if use_gpu else "💻 CPU Mode")

    def to_tensor(self, img):
        return torch.from_numpy(img).to(self.torch_device) if self.use_gpu else img

    def to_numpy(self, tensor):
        return tensor.cpu().numpy() if self.use_gpu and isinstance(tensor, torch.Tensor) else tensor

    def gaussian_blur_gpu(self, img, kernel_size=(9, 9), sigma=2):
        if not self.use_gpu:
            return cv2.GaussianBlur(img, kernel_size, sigma)

        img_tensor = self.to_tensor(img.astype(np.float32))
        kernel_1d = self._gaussian_kernel_1d(kernel_size[0], sigma).to(self.torch_device)

        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)

        img_tensor = torch.nn.functional.conv2d(
            img_tensor, kernel_1d.view(1, 1, 1, -1),
            padding=(0, kernel_size[0]//2)
        )
        img_tensor = torch.nn.functional.conv2d(
            img_tensor, kernel_1d.view(1, 1, -1, 1),
            padding=(kernel_size[0]//2, 0)
        )

        return self.to_numpy(img_tensor.squeeze().clamp(0,255).byte())

    def _gaussian_kernel_1d(self, size, sigma):
        coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
        kernel = torch.exp(-(coords**2) / (2 * sigma**2))
        return kernel / kernel.sum()

    def morphology_gpu(self, img, operation, kernel_size=5, iterations=1):
        if not self.use_gpu:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size,kernel_size))
            if operation == 'close': return cv2.morphologyEx(img, cv2.MORPH_CLOSE, k, iterations=iterations)
            if operation == 'open': return cv2.morphologyEx(img, cv2.MORPH_OPEN, k, iterations=iterations)
            return img

        img_tensor = self.to_tensor(img.astype(np.float32)/255.0)
        kernel = torch.ones(1,1,kernel_size,kernel_size).to(self.torch_device)

        for _ in range(iterations):
            t = img_tensor.unsqueeze(0).unsqueeze(0)
            if operation in ['dilate','close']:
                img_tensor = torch.nn.functional.max_pool2d(t, kernel_size, stride=1, padding=kernel_size//2).squeeze()
            elif operation in ['erode','open']:
                img_tensor = -torch.nn.functional.max_pool2d(-t, kernel_size, stride=1, padding=kernel_size//2).squeeze()

        return (img_tensor*255).clamp(0,255).byte().cpu().numpy()

    def threshold_gpu(self, img):
        return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# ==================== CORE PROCESSING ====================
def clean_and_thin_skeleton(skel_img, close_kernel_size=3, prune_iters=8):
    sk = (skel_img > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel_size,close_kernel_size))
    closed = cv2.morphologyEx(sk*255, cv2.MORPH_CLOSE, kernel)
    thin = skeletonize(closed//255).astype(np.uint8)

    neigh = np.array([[1,1,1],[1,0,1],[1,1,1]], np.uint8)
    pruned = thin.copy()
    for _ in range(prune_iters):
        neigh_count = cv2.filter2D(pruned, -1, neigh)
        endpoints = (pruned==1) & (neigh_count==1)
        if not endpoints.any(): break
        pruned[endpoints] = 0
    return pruned

def find_vein_junctions(skeleton_img, min_degree=3):
    sk_clean = clean_and_thin_skeleton(skeleton_img)
    kernel = np.ones((3,3), np.uint8)
    kernel[1,1] = 0
    neigh_count = cv2.filter2D(sk_clean, -1, kernel)
    mask = (sk_clean==1) & (neigh_count>=min_degree)
    ys, xs = np.nonzero(mask)
    return list(zip(xs,ys)), sk_clean

def cluster_points_by_components(points, image_shape, cluster_radius=12):
    if len(points)==0: return []
    canvas = np.zeros(image_shape, np.uint8)
    for x,y in points:
        cv2.circle(canvas,(int(x),int(y)),cluster_radius,255,-1)

    num_labels, labels = cv2.connectedComponents(canvas)
    final = []
    for lab in range(1, num_labels):
        ys, xs = np.nonzero(labels==lab)
        if len(xs)==0: continue
        final.append((int(xs.mean()), int(ys.mean())))
    return final

# ==================== PROCESS ONE IMAGE ====================
def process_single_image(image_path, output_dir, gpu_processor, save_visualizations=True):
    img_name = Path(image_path).stem
    out_dir = os.path.join(output_dir, img_name)
    os.makedirs(out_dir, exist_ok=True)

    img_color = cv2.imread(str(image_path))
    if img_color is None:
        return None, f"Cannot load image {image_path}"

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    h,w = img_gray.shape

    clahe = cv2.createCLAHE(clipLimit=3,tileGridSize=(8,8))
    enhanced = clahe.apply(img_gray)

    thresh = gpu_processor.threshold_gpu(enhanced)
    if np.mean(img_gray[thresh==255]) > np.mean(img_gray[thresh==0]):
        thresh = cv2.bitwise_not(thresh)

    thresh = gpu_processor.morphology_gpu(thresh,'close',iterations=2)
    thresh = gpu_processor.morphology_gpu(thresh,'open',iterations=2)

    contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    wing_mask = np.zeros_like(img_gray)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(wing_mask,[largest],-1,255,-1)
        selected_contour = largest
    else:
        wing_mask[:]=255
        selected_contour=None

    masked = cv2.bitwise_and(enhanced, wing_mask)

    # Thick morphology branch
    inverted = cv2.bitwise_not(masked)
    blurred = gpu_processor.gaussian_blur_gpu(inverted)
    thick_binary = gpu_processor.threshold_gpu(blurred)

    thick_bin_labeled = thick_binary.astype(bool)
    thick_clean = remove_small_objects(thick_bin_labeled, min_size=500)
    thick_clean = (thick_clean*255).astype(np.uint8)

    thick_clean = cv2.morphologyEx(thick_clean, cv2.MORPH_CLOSE,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)),2)
    thick_clean = cv2.bitwise_and(thick_clean, wing_mask)

    thick_skel = (skeletonize(thick_clean//255)*255).astype(np.uint8)
    thick_skel = cv2.bitwise_and(thick_skel, wing_mask)

    # Frangi branch
    masked_norm = masked.astype(np.float64)/255.0
    fr = frangi(masked_norm, sigmas=range(3,10), black_ridges=True)
    fr = (fr*255).astype(np.uint8)

    _, fr_bin = cv2.threshold(fr, 50,255,cv2.THRESH_BINARY)
    fr_bin = cv2.bitwise_and(fr_bin, wing_mask)

    fr_clean = remove_small_objects(fr_bin.astype(bool), min_size=300)
    fr_clean = (fr_clean*255).astype(np.uint8)
    fr_clean = cv2.bitwise_and(fr_clean, wing_mask)

    fr_skel = (skeletonize(fr_clean//255)*255).astype(np.uint8)
    fr_skel = cv2.bitwise_and(fr_skel, wing_mask)

    # Combine skeletons
    vein_skel = cv2.bitwise_or(thick_skel, fr_skel)
    vein_skel = cv2.bitwise_and(vein_skel, wing_mask)

    # ==================== JUNCTION DETECTION ====================
    raw_points, _ = find_vein_junctions(vein_skel, min_degree=3)

    # ✅ ✅ ✅ CLUSTER POINTS TO REMOVE NEARBY DUPLICATES
    CLUSTER_RADIUS = 12  
    vein_junctions = cluster_points_by_components(raw_points, vein_skel.shape, CLUSTER_RADIUS)

    # ==================== SAVE RESULTS ====================
    result = {
        'image_name': img_name,
        'image_path': str(image_path),
        'width': w,
        'height': h,
        'vein_junctions': vein_junctions,
        'num_vein_junctions': len(vein_junctions)
    }

    if save_visualizations:
        ann = img_color.copy()
        overlay = cv2.cvtColor(vein_skel, cv2.COLOR_GRAY2BGR)
        overlay[vein_skel>0] = [200,200,200]
        ann = cv2.addWeighted(ann,1.0,overlay,0.4,0)

        for x,y in vein_junctions:
            cv2.circle(ann,(x,y),8,(0,0,255),-1)
            cv2.circle(ann,(x,y),10,(0,255,255),2)

        if selected_contour is not None:
            cv2.drawContours(ann,[selected_contour],-1,(0,255,0),2)

        cv2.imwrite(os.path.join(out_dir,"annotated_image.jpg"), ann)
        cv2.imwrite(os.path.join(out_dir,"vein_skeleton.jpg"), vein_skel)
        shutil.copy(image_path, os.path.join(out_dir,"original_image.jpg"))

    # CSV
    with open(os.path.join(out_dir,"vein_junctions.csv"),'w') as f:
        f.write("ID,X,Y\n")
        for i,(x,y) in enumerate(vein_junctions,1):
            f.write(f"{i},{x},{y}\n")

    # JSON
    with open(os.path.join(out_dir,"annotations.json"),'w') as f:
        json.dump({
            'image': img_name,
            'width': w,
            'height': h,
            'junctions': [{'id':i+1,'x':x,'y':y} for i,(x,y) in enumerate(vein_junctions)]
        }, f, indent=2)

    # YOLO (pose format)
    KEYPOINT_VISIBILITY = 2
    with open(os.path.join(out_dir,"yolo_annotations.txt"),'w') as f:
        for x,y in vein_junctions:
            xn = x/w
            yn = y/h
            bw, bh = 16/w, 16/h
            f.write(f"0 {xn:.6f} {yn:.6f} {bw:.6f} {bh:.6f} {xn:.6f} {yn:.6f} {KEYPOINT_VISIBILITY}\n")

    return result, None

# ==================== BATCH PROCESSING ====================
def process_dataset(input_folder, output_base_dir="./results", save_visualizations=True, use_gpu=True):
    gpu_available, dev = setup_gpu()
    if not gpu_available: use_gpu=False
    gpu_proc = GPUImageProcessor(use_gpu, dev)

    folder = Path(input_folder)
    if not folder.exists():
        print("❌ Folder does not exist.")
        return None, None, None, None

    # Load images once, no duplicates
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    imgs = [p for p in folder.iterdir() if p.suffix.lower() in exts]

    if not imgs:
        print("❌ No images found.")
        return None, None, None, None

    print(f"✅ Found {len(imgs)} unique images")

    if not imgs:
        print("❌ No images found.")
        return None,None,None,None

    print(f"✅ Found {len(imgs)} images")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(output_base_dir, f"batch_{timestamp}")
    os.makedirs(out, exist_ok=True)

    all_results=[]
    errors=[]

    for p in tqdm(imgs):
        r, e = process_single_image(p, out, gpu_proc, save_visualizations)
        if e: errors.append({'image':str(p),'error':e})
        else: all_results.append(r)

        if use_gpu and len(all_results)%10==0:
            clear_gpu_memory()

    df = pd.DataFrame([{
        'Image': r['image_name'],
        'Width': r['width'],
        'Height': r['height'],
        'Vein_Junctions': r['num_vein_junctions']
    } for r in all_results])

    df.to_csv(os.path.join(out,"dataset_summary.csv"),index=False)

    return out, df, all_results, errors

# ==================== MAIN ====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i","--input",required=True)
    ap.add_argument("-o","--output",default="./results")
    ap.add_argument("--cpu",action="store_true")
    ap.add_argument("--no-viz",action="store_true")
    args = ap.parse_args()

    gpu_available,_ = setup_gpu()
    use_gpu = gpu_available and not args.cpu

    out,df,res,err = process_dataset(
        args.input,
        args.output,
        save_visualizations=not args.no_viz,
        use_gpu=use_gpu
    )

    if out:
        print(f"✅ FINISHED — Results saved to: {out}")
    else:
        print("❌ FAILED")

if __name__ == "__main__":
    main()
