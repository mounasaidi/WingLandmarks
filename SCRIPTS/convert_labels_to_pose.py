"""
Convert 5-column YOLO labels to 7-column YOLOv8 Pose format
Run this BEFORE training to convert all annotation files
"""

import os
from pathlib import Path

# ==================== CONFIGURATION ====================
RESULTS_BASE_DIR = './results'
KEYPOINT_VISIBILITY_FLAG = 2  # 2 means 'visible and labeled'

# Find the latest batch directory
batch_dirs = sorted([d for d in Path(RESULTS_BASE_DIR).glob('batch_*') if d.is_dir()], 
                   key=lambda p: p.name, reverse=True)

if not batch_dirs:
    print(f"❌ No batch directories found in {RESULTS_BASE_DIR}")
    exit(1)

LATEST_BATCH_DIR = batch_dirs[0]

# ==================== CONVERSION FUNCTION ====================
def convert_single_label_file(label_path: Path) -> bool:
    """
    Convert a single label file from 5-column to 7-column format.
    Returns True if successful, False otherwise.
    """
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            print(f"⚠️  Empty file: {label_path.name}")
            return False
        
        converted_lines = []
        
        for i, line in enumerate(lines):
            parts = line.strip().split()
            
            if len(parts) == 5:
                # Standard 5-column format: class_id x_center y_center width height
                class_id, x_center, y_center, width, height = parts
                
                # Convert to YOLOv8 pose: class_id x_c y_c w h keypoint_x keypoint_y visibility
                # Use the bounding box center as the keypoint location
                new_line = f"{class_id} {x_center} {y_center} {width} {height} {x_center} {y_center} {KEYPOINT_VISIBILITY_FLAG}\n"
                converted_lines.append(new_line)
                
            elif len(parts) == 8:
                # 8-column format: class_id x_c y_c w h kp_x kp_y visibility
                # This IS the correct YOLOv8 pose format for 1 keypoint!
                print(f"✓ Already in correct YOLOv8 pose format: {label_path.name}")
                return True
                
            elif len(parts) > 0:
                print(f"❌ Invalid format in {label_path.name}, line {i+1}: {len(parts)} columns")
                print(f"   Data: {parts}")
                return False
        
        if converted_lines:
            # Write the converted content back to the file
            with open(label_path, 'w') as f:
                f.writelines(converted_lines)
            return True
        else:
            print(f"⚠️  No valid lines in: {label_path.name}")
            return False
            
    except Exception as e:
        print(f"❌ Error processing {label_path.name}: {str(e)}")
        return False


# ==================== MAIN CONVERSION ====================
def convert_all_labels():
    """Find and convert all label files in the batch directory."""
    
    print(f"\n🔍 Searching for label files in: {LATEST_BATCH_DIR}")
    print("="*60)
    
    # Find all yolo_annotations.txt files
    label_files = list(LATEST_BATCH_DIR.glob('*/yolo_annotations.txt'))
    
    if not label_files:
        print("❌ No label files found!")
        return
    
    print(f"📁 Found {len(label_files)} label files\n")
    
    success_count = 0
    failed_count = 0
    
    # Process each label file
    print("Converting labels...")
    for idx, label_file in enumerate(label_files, 1):
        print(f"[{idx}/{len(label_files)}] {label_file.name}", end=" ... ")
        if convert_single_label_file(label_file):
            print("✓")
            success_count += 1
        else:
            print("✗")
            failed_count += 1
    
    # Summary
    print("\n" + "="*60)
    print("✅ CONVERSION COMPLETE!")
    print("="*60)
    print(f"✓ Successfully converted: {success_count} files")
    print(f"✗ Failed: {failed_count} files")
    print(f"📊 Total: {len(label_files)} files")
    print("="*60)
    
    if failed_count > 0:
        print("\n⚠️  Some files failed to convert. Check the errors above.")
        print("   You may need to manually fix these files.")
    else:
        print("\n🎉 All labels successfully converted!")
        print("   You can now run the training script.")


if __name__ == '__main__':
    convert_all_labels()