import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import shutil
def simulate_low_light(image, factor=0.3):
    """Simulate low-light image by scaling brightness."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[..., 2] = (hsv[..., 2] * factor).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
def prepare_dataset(input_dir, output_dir, val_ratio=0.1, test_ratio=0.2, image_size=(224, 224)):
    # Step 1: Read all images
    image_list = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_list.sort()  # optional, for consistent order
    if len(image_list) == 0:
        print("No images found in", input_dir)
        return
    # Step 2: Train/Val/Test split
    trainval_imgs, test_imgs = train_test_split(image_list, test_size=test_ratio, random_state=42)
    train_imgs, val_imgs = train_test_split(trainval_imgs, test_size=val_ratio / (1 - test_ratio), random_state=42)
    splits = {
        'train': train_imgs,
        'val': val_imgs,
        'test': test_imgs
    }
    # Step 3: Process and save
    for split, filenames in splits.items():
        print(f"Processing {split} split with {len(filenames)} images...")
        low_dir = os.path.join(output_dir, split, 'low')
        high_dir = os.path.join(output_dir, split, 'high')
        os.makedirs(low_dir, exist_ok=True)
        os.makedirs(high_dir, exist_ok=True)
        for fname in tqdm(filenames):
            image_path = os.path.join(input_dir, fname)
            image = cv2.imread(image_path)
            if image is None:
                print("⚠️ Skipping unreadable:", fname)
                continue
            # Resize to desired size
            image_resized = cv2.resize(image, image_size)
            low_light_img = simulate_low_light(image_resized)

            cv2.imwrite(os.path.join(high_dir, fname), image_resized)
            cv2.imwrite(os.path.join(low_dir, fname), low_light_img)
    print("✅ Dataset split and low-light generation complete!")
# Paths
original_dataset_path = "/content/cvccolondb/data/train/images"
output_dataset_path = "/content/cvccolondbsplit"

# Run with resizing to 224x224
prepare_dataset(original_dataset_path, output_dataset_path, image_size=(224, 224))
