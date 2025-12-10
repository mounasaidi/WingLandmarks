"""
Quick Test - Test model on a single image with keypoints only
"""

import cv2
from ultralytics import YOLO
from pathlib import Path

# Configuration
MODEL_PATH = './runs/pose/tsetse_wing_landmarks_full_batch17/weights/best.pt'
IMAGE_PATH = 'C:/Users/rayen/OneDrive/Desktop/finalact/tsetse_wing_yolov8/data/images/train/A001___20170126_221605.jpg'
OUTPUT_PATH = './quick_test_result_points_only3.jpg'


def quick_test(model_path, image_path, output_path):
    """Quick test on a single image – draws only keypoints"""

    # Check files
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return

    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return

    print("📦 Loading model...")
    model = YOLO(model_path)

    print(f"🔍 Running inference on: {image_path}")
    results = model.predict(
        source=image_path,
        conf=0.3,
        save=False,
        device=0,
        show=False
    )

    result = results[0]

    # Load original image
    img = cv2.imread(image_path)

    print("\n📊 Detection Details:")
    num_detections = len(result.keypoints) if result.keypoints is not None else 0
    print(f"✅ Found {num_detections} keypoint sets")

    # ---------------------------------------------------------
    # 🔵 Draw ONLY keypoints (no boxes)
    # ---------------------------------------------------------
    if result.keypoints is not None:
        for i, kpts in enumerate(result.keypoints):
            pts = kpts.xy.cpu().numpy()

            # If shape is (1, N, 2) → reduce it
            if pts.ndim == 3:
                pts = pts[0]

            # If shape is flat → reshape
            if pts.ndim == 1:
                pts = pts.reshape(-1, 2)

            # Draw each point
            for p in pts:
                x, y = int(p[0]), int(p[1])
                cv2.circle(img, (x, y), radius=5, color=(255, 0, 0), thickness=-1)


            # Confidence printout
            if result.boxes is not None:
                conf = float(result.boxes[i].conf[0])
                print(f"   Point Set {i+1}: Confidence = {conf:.3f}")

    # ---------------------------------------------------------

    # Save image
    cv2.imwrite(output_path, img)
    print(f"💾 Saved keypoints-only result to: {output_path}")

    print(f"\n✨ Open '{output_path}' to see the keypoints-only visualization!")


if __name__ == "__main__":
    quick_test(MODEL_PATH, IMAGE_PATH, OUTPUT_PATH)
