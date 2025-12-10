import os
import shutil
from pathlib import Path
from ultralytics import YOLO

# ==================== CONFIGURATION (CHANGED MODEL) ====================
# NOTE: Set the path to the folder containing your 'batch_*' subdirectories
RESULTS_BASE_DIR = './results' 
# Find the latest batch directory dynamically
batch_dirs = sorted([d for d in Path(RESULTS_BASE_DIR).glob('batch_*') if d.is_dir()], key=lambda p: p.name, reverse=True)
if not batch_dirs:
    print(f"❌ No batch directories found in {RESULTS_BASE_DIR}")
    LATEST_BATCH_DIR = None 
else:
    LATEST_BATCH_DIR = batch_dirs[0] 

PROJECT_ROOT = Path('./tsetse_wing_yolov8')
DATA_ROOT = PROJECT_ROOT / 'data'
# *** MODEL CHANGED HERE ***
MODEL_TYPE = 'yolov8s-pose.pt' # YOLOv8 Small for Keypoint Estimation (Improved Accuracy)
# *************************
EPOCHS = 50 
BATCH_SIZE = 16 
CLASS_NAMES = ['vein_junction']
KEYPOINT_VISIBILITY_FLAG = 2 # 2 means 'visible and labeled'

# ==================== LABEL CONVERSION LOGIC (DIAGNOSTIC) ====================
def convert_label_to_pose_format(ann_path: Path):
    """
    Reads a 5-column YOLO file and returns the 7-column YOLOv8-Pose content.
    Includes print statements for diagnosing unexpected label formats.
    """
    
    print(f"\n🔬 Checking file: {ann_path.name}")
    with open(ann_path, 'r') as f:
        lines = f.readlines()
        
    new_lines = []
    
    for i, line in enumerate(lines):
        parts = line.strip().split()
        
        # Check for 5 columns: class_id x_c y_c w h
        if len(parts) == 5:
            # --- CONVERSION LOGIC ---
            class_id = parts[0]
            x_center = parts[1]
            y_center = parts[2]
            box_w = parts[3]
            box_h = parts[4]
            keypoint_x = x_center
            keypoint_y = y_center
            
            new_line = (
                f"{class_id} {x_center} {y_center} {box_w} {box_h} "
                f"{keypoint_x} {keypoint_y} {KEYPOINT_VISIBILITY_FLAG}\n"
            )
            new_lines.append(new_line)
        elif len(parts) >= 7:
            print(f"✅ File {ann_path.name} already in Pose format. Skipping conversion.")
            return None 
        elif len(parts) > 0:
            # CRITICAL DIAGNOSTIC: This line runs if it's NOT 5 columns and NOT 7+ columns.
            print(f"❌ ERROR: Line {i+1} in {ann_path.name} has {len(parts)} columns (Expected 5). Data: {parts}")
            return None
        else:
            # Ignore empty lines
            pass
            
    if not new_lines and len(lines) > 0:
        print(f"🛑 Conversion failed for {ann_path.name}. All lines were corrupted.")
        return None
    
    if new_lines:
        print(f"✅ Successfully converted {len(new_lines)} lines in {ann_path.name}.")
        print(f"   Example: {new_lines[0].strip()}")
        return new_lines
    
    return None

# ==================== DATASET PREPARATION (MODIFIED) ====================
def prepare_full_dataset(latest_batch_dir: Path):
    """
    Gathers all image/annotation paths, converts 5-column labels to 7-column Pose format, 
    and copies them entirely to the 'train' folder.
    """
    if not latest_batch_dir: return

    print(f"Gathering data from: {latest_batch_dir}")
    
    image_paths = []
    annotation_paths = []

    # Traverse the batch directory structure to find all processed samples
    for sample_dir in latest_batch_dir.glob('*/'):
        if sample_dir.is_dir():
            original_img = sample_dir / 'original_image.jpg'
            yolo_ann = sample_dir / 'yolo_annotations.txt'
            
            if original_img.exists() and yolo_ann.exists() and original_img.stat().st_size > 0:
                image_paths.append(original_img)
                annotation_paths.append(yolo_ann)

    if not image_paths:
        print("❌ No complete (image + yolo annotation) samples found.")
        return

    print(f"✅ Found {len(image_paths)} samples in total. Preparing as single 'train' set.")
    
    # Create the single training structure
    train_images_dir = DATA_ROOT / 'images' / 'train'
    train_labels_dir = DATA_ROOT / 'labels' / 'train'
    
    # Clear and create directories (CRUCIAL for removing old cache/bad files!)
    if DATA_ROOT.exists():
        shutil.rmtree(DATA_ROOT)
        
    train_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    # Copy all files and convert labels on the fly
    for img_path, ann_path in zip(image_paths, annotation_paths):
        # 1. Convert label file content
        pose_content = convert_label_to_pose_format(ann_path)
        
        is_converted = pose_content is not None
        
        # 2. Create a unique file name
        final_file_name = f"{img_path.parts[-2].replace(' ', '_').replace('-', '_')}"
        final_img_name = f"{final_file_name}.jpg"
        final_ann_name = f"{final_file_name}.txt"
        
        # 3. Copy image
        shutil.copy(img_path, train_images_dir / final_img_name)
        
        # 4. Write the converted label file or copy the original
        if is_converted:
            # Write the converted 7-column content
            with open(train_labels_dir / final_ann_name, 'w') as f:
                f.writelines(pose_content)
        else:
            # Copy the original file (if it was already 7+ columns, or if conversion failed)
            shutil.copy(ann_path, train_labels_dir / final_ann_name)
            
        processed_count += 1


    print(f"\n✅ Successfully processed and copied {processed_count} samples to the training directories.")


# ==================== YOLO CONFIGURATION AND TRAINING ====================
def create_yaml_config():
    """Creates the necessary data.yaml file for YOLOv8."""
    yaml_content = f"""
# YOLOv8 custom training data configuration
path: {str(PROJECT_ROOT.absolute() / 'data')}  # Dataset root path
train: images/train    # Train images path relative to 'path'
val: images/train      # Val images path (using train for now)

# Classes
nc: {len(CLASS_NAMES)}  # Number of classes
names: {CLASS_NAMES}    # Class names

# Keypoints - IMPORTANT: 1 keypoint with 3 values (x, y, visibility)
kpt_shape: [1, 3]  # Changed from [1, 2] to [1, 3]
    """
    yaml_path = PROJECT_ROOT / 'tsetse_data_full.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"✅ Configuration file created at: {yaml_path}")
    return yaml_path

def train_yolov8_model(data_yaml_path):
    """Initializes and trains the YOLOv8-pose model."""
    print("\n--- Starting YOLOv8 Keypoint Training ---")
    
    # Load a pretrained YOLO-Pose model (now small size)
    model = YOLO(MODEL_TYPE) 
    
    # Train the model
    results = model.train(
    data=str(data_yaml_path),
    epochs=EPOCHS,
    imgsz=640, 
    batch=BATCH_SIZE,
    name='tsetse_wing_landmarks_full_batch',
    task='pose',
    device='0'  # Add this line! '0' = use GPU 0
)
    print("\n--- Training Complete ---")
    print(f"Results saved in: {model.trainer.save_dir}")
    return results

if __name__ == '__main__':
    if LATEST_BATCH_DIR:
        # 1. Prepare Data (Conversion included)
        prepare_full_dataset(LATEST_BATCH_DIR)
        
        # 2. Create YAML Config
        yaml_path = create_yaml_config()
        
        # 3. Train Model
        train_yolov8_model(yaml_path)
    else:
        print("Training aborted because no batch data directory was found.")