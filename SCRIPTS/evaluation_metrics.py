"""
Evaluate Model Performance
Get detailed metrics: mAP, precision, recall, etc.
"""

from ultralytics import YOLO
from pathlib import Path
import json

# Configuration
MODEL_PATH = './runs/pose/tsetse_wing_landmarks_full_batch17/weights/best.pt'
DATA_YAML = './tsetse_wing_yolov8/tsetse_data_full.yaml'

def evaluate_model(model_path, data_yaml):
    """
    Evaluate model on validation/test set
    """
    print("🦋 MODEL EVALUATION")
    print("="*60)
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    # Load model
    print(f"📦 Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Run validation
    print(f"\n🔍 Running validation...")
    results = model.val(
        data=data_yaml,
        device=0,  # Use GPU
        plots=True
    )
    
    # Print results
    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS")
    print("="*60)
    
    # Box metrics
    print("\n📦 Bounding Box Detection:")
    print(f"   Precision: {results.box.mp:.3f}")
    print(f"   Recall: {results.box.mr:.3f}")
    print(f"   mAP50: {results.box.map50:.3f}")
    print(f"   mAP50-95: {results.box.map:.3f}")
    
    # Pose metrics
    print("\n🎯 Keypoint Detection:")
    print(f"   Precision: {results.pose.mp:.3f}")
    print(f"   Recall: {results.pose.mr:.3f}")
    print(f"   mAP50: {results.pose.map50:.3f}")
    print(f"   mAP50-95: {results.pose.map:.3f}")
    
    # Speed
    print(f"\n⚡ Speed:")
    print(f"   Preprocess: {results.speed['preprocess']:.1f}ms")
    print(f"   Inference: {results.speed['inference']:.1f}ms")
    print(f"   Postprocess: {results.speed['postprocess']:.1f}ms")
    print(f"   Total: {sum(results.speed.values()):.1f}ms per image")
    
    print("\n" + "="*60)
    
    # Save metrics to file
    metrics_path = Path(model_path).parent.parent / 'evaluation_metrics.json'
    metrics = {
        'box': {
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
            'mAP50': float(results.box.map50),
            'mAP50-95': float(results.box.map)
        },
        'pose': {
            'precision': float(results.pose.mp),
            'recall': float(results.pose.mr),
            'mAP50': float(results.pose.map50),
            'mAP50-95': float(results.pose.map)
        },
        'speed_ms': {
            'preprocess': results.speed['preprocess'],
            'inference': results.speed['inference'],
            'postprocess': results.speed['postprocess'],
            'total': sum(results.speed.values())
        }
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"💾 Metrics saved to: {metrics_path}")
    
    return results

if __name__ == "__main__":
    evaluate_model(MODEL_PATH, DATA_YAML)