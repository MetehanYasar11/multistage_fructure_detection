"""
Check actual image-level class distribution
"""
import json
import numpy as np

# Load split
with open('outputs/splits/train_val_test_split.json', 'r') as f:
    splits = json.load(f)

train_images = splits['train']

print("="*80)
print("ACTUAL IMAGE-LEVEL CLASS DISTRIBUTION")
print("="*80)

# Count unique images per class
fractured_images = set()
healthy_images = set()

for item in train_images:
    image_id = item['image_id']
    label = item['label']
    
    if label == 1:
        fractured_images.add(image_id)
    else:
        healthy_images.add(image_id)

print(f"\nTrain split:")
print(f"  Unique FRACTURED images: {len(fractured_images)}")
print(f"  Unique HEALTHY images: {len(healthy_images)}")
print(f"  Total unique images: {len(fractured_images | healthy_images)}")

# Count labels per image
image_label_counts = {}
for item in train_images:
    image_id = item['image_id']
    if image_id not in image_label_counts:
        image_label_counts[image_id] = {'fractured': 0, 'healthy': 0}
    
    if item['label'] == 1:
        image_label_counts[image_id]['fractured'] += 1
    else:
        image_label_counts[image_id]['healthy'] += 1

print(f"\n" + "="*80)
print("LABELS PER IMAGE STATISTICS:")
print("="*80)

fractured_label_counts = [counts['fractured'] for img_id, counts in image_label_counts.items() 
                          if img_id in fractured_images]
healthy_label_counts = [counts['healthy'] for img_id, counts in image_label_counts.items() 
                        if img_id in healthy_images]

print(f"\nFractured images:")
print(f"  Labels per image - mean: {np.mean(fractured_label_counts):.2f}")
print(f"  Labels per image - min: {np.min(fractured_label_counts)}")
print(f"  Labels per image - max: {np.max(fractured_label_counts)}")

print(f"\nHealthy images:")
print(f"  Labels per image - mean: {np.mean(healthy_label_counts):.2f}")
print(f"  Labels per image - min: {np.min(healthy_label_counts)}")
print(f"  Labels per image - max: {np.max(healthy_label_counts)}")

print(f"\n" + "="*80)
print("ACTUAL IMBALANCE:")
print("="*80)
ratio = len(fractured_images) / len(healthy_images)
print(f"Fractured/Healthy ratio: {ratio:.2f}")
if ratio > 1:
    print(f"→ Dataset has {ratio:.2f}x MORE fractured images")
    print(f"→ Should INCREASE weight for HEALTHY class in loss")
else:
    print(f"→ Dataset has {1/ratio:.2f}x MORE healthy images")
    print(f"→ Should INCREASE weight for FRACTURED class in loss")

# Show what dataset.py sees
print(f"\n" + "="*80)
print("WHAT DATASET SEES (WRONG!):")
print("="*80)
print(f"Total train items: {len(train_images)}")
fractured_count = sum(1 for item in train_images if item['label'] == 1)
healthy_count = sum(1 for item in train_images if item['label'] == 0)
print(f"Fractured labels: {fractured_count}")
print(f"Healthy labels: {healthy_count}")
print(f"→ This is MISLEADING because healthy images have multiple labels!")
