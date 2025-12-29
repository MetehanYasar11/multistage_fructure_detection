"""
Auto-label new crops using ground truth and merge with existing training data

Strategy:
- Load crop-level split (75% train / 25% val)
- ONLY use TRAIN crops for merging with existing training data
- Keep VAL crops separate for validation
"""

import json
import shutil
from pathlib import Path
from tqdm import tqdm

# Paths
NEW_CROPS_DIR = Path('new_data_crops_train_sr_clahe')
NEW_CROPS_SPLIT = Path('new_data_crops_split.json')
ORIGINAL_TRAINING_DATA = Path('auto_labeled_crops_sr_clahe')
OUTPUT_DIR = Path('auto_labeled_crops_sr_clahe_FINAL')
VAL_OUTPUT_DIR = Path('new_data_crops_val_sr_clahe')


def load_crop_split():
    """Load crop-level train/val split"""
    print("\n📂 Loading crop split...")
    
    with open(NEW_CROPS_SPLIT, 'r') as f:
        split_data = json.load(f)
    
    train_fractured = split_data['train']['fractured']
    train_healthy = split_data['train']['healthy']
    val_fractured = split_data['val']['fractured']
    val_healthy = split_data['val']['healthy']
    
    print(f"   Training crops: {split_data['train']['total']}")
    print(f"      - Fractured: {len(train_fractured)}")
    print(f"      - Healthy: {len(train_healthy)}")
    
    print(f"   Validation crops: {split_data['val']['total']}")
    print(f"      - Fractured: {len(val_fractured)}")
    print(f"      - Healthy: {len(val_healthy)}")
    
    return {
        'train': {'fractured': train_fractured, 'healthy': train_healthy},
        'val': {'fractured': val_fractured, 'healthy': val_healthy}
    }


def organize_train_crops(train_crops):
    """Organize training crops into healthy/ and fractured/ folders"""
    print("\n📁 Organizing TRAINING crops by label...")
    
    # Create temporary output directory
    temp_dir = Path('new_data_crops_train_labeled')
    healthy_dir = temp_dir / 'healthy'
    fractured_dir = temp_dir / 'fractured'
    
    healthy_dir.mkdir(parents=True, exist_ok=True)
    fractured_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy fractured crops
    for crop_name in tqdm(train_crops['fractured'], desc="Fractured"):
        src_path = NEW_CROPS_DIR / crop_name
        if src_path.exists():
            shutil.copy2(src_path, fractured_dir / crop_name)
    
    # Copy healthy crops
    for crop_name in tqdm(train_crops['healthy'], desc="Healthy"):
        src_path = NEW_CROPS_DIR / crop_name
        if src_path.exists():
            shutil.copy2(src_path, healthy_dir / crop_name)
    
    print(f"   ✅ Organized {len(train_crops['healthy'])} healthy + {len(train_crops['fractured'])} fractured crops")
    
    return temp_dir


def organize_val_crops(val_crops):
    """Organize validation crops into separate directory"""
    print("\n📁 Organizing VALIDATION crops...")
    
    # Create validation directory
    VAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    healthy_dir = VAL_OUTPUT_DIR / 'healthy'
    fractured_dir = VAL_OUTPUT_DIR / 'fractured'
    
    healthy_dir.mkdir(exist_ok=True)
    fractured_dir.mkdir(exist_ok=True)
    
    # Copy fractured crops
    for crop_name in tqdm(val_crops['fractured'], desc="Fractured"):
        src_path = NEW_CROPS_DIR / crop_name
        if src_path.exists():
            shutil.copy2(src_path, fractured_dir / crop_name)
    
    # Copy healthy crops
    for crop_name in tqdm(val_crops['healthy'], desc="Healthy"):
        src_path = NEW_CROPS_DIR / crop_name
        if src_path.exists():
            shutil.copy2(src_path, healthy_dir / crop_name)
    
    print(f"   ✅ Validation crops saved to: {VAL_OUTPUT_DIR}")
    print(f"      - Healthy: {len(val_crops['healthy'])}")
    print(f"      - Fractured: {len(val_crops['fractured'])}")


def merge_with_original_dataset(new_crops_dir):
    """
    Merge new crops with original auto_labeled_crops_sr_clahe dataset
    
    Args:
        new_crops_dir: Path to new labeled crops (with healthy/ and fractured/ subdirs)
    """
    print("\n🔗 Merging with original training dataset...")
    
    # Create final output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Copy original training data
    print("   Copying original training data...")
    for class_name in ['healthy', 'fractured']:
        src_dir = ORIGINAL_TRAINING_DATA / class_name
        dst_dir = OUTPUT_DIR / class_name
        
        dst_dir.mkdir(exist_ok=True)
        
        if src_dir.exists():
            crops = list(src_dir.glob('*.jpg')) + list(src_dir.glob('*.png'))
            for crop in tqdm(crops, desc=f"   {class_name}", leave=False):
                shutil.copy2(crop, dst_dir / crop.name)
    
    # Add new crops with "new_" prefix to avoid conflicts
    print("   Adding new crops...")
    for class_name in ['healthy', 'fractured']:
        src_dir = new_crops_dir / class_name
        dst_dir = OUTPUT_DIR / class_name
        
        if src_dir.exists():
            crops = list(src_dir.glob('*.jpg')) + list(src_dir.glob('*.png'))
            for crop in tqdm(crops, desc=f"   new_{class_name}", leave=False):
                # Add "new_" prefix to distinguish from original crops
                new_name = f"new_{crop.name}"
                shutil.copy2(crop, dst_dir / new_name)
    
    # Count final dataset
    healthy_count = len(list((OUTPUT_DIR / 'healthy').glob('*')))
    fractured_count = len(list((OUTPUT_DIR / 'fractured').glob('*')))
    total_count = healthy_count + fractured_count
    
    print(f"\n   ✅ Final dataset created: {OUTPUT_DIR}")
    print(f"   Total: {total_count} crops")
    print(f"   - Healthy: {healthy_count}")
    print(f"   - Fractured: {fractured_count}")
    
    # Save statistics
    stats = {
        'total_crops': total_count,
        'healthy': healthy_count,
        'fractured': fractured_count,
        'original_dataset': str(ORIGINAL_TRAINING_DATA),
        'new_crops_added': len(list(new_crops_dir.rglob('*.jpg')))
    }
    
    stats_path = OUTPUT_DIR / 'dataset_statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"   ✅ Statistics saved: {stats_path}")


def main():
    print("="*80)
    print("🤖 AUTO-LABELING & MERGING NEW TRAINING DATA")
    print("="*80)
    
    # Load crop split (train/val)
    crop_split = load_crop_split()
    
    # Organize training crops by label
    train_crops_dir = organize_train_crops(crop_split['train'])
    
    # Organize validation crops
    organize_val_crops(crop_split['val'])
    
    # Merge training crops with original dataset
    merge_with_original_dataset(train_crops_dir)
    
    print("\n" + "="*80)
    print("✅ AUTO-LABELING & MERGING COMPLETED!")
    print("="*80)
    print(f"\n📊 Final training dataset: {OUTPUT_DIR}")
    print(f"   - Original: {ORIGINAL_TRAINING_DATA}")
    print(f"   - New crops (train only): {train_crops_dir}")
    print(f"   - Merged output: {OUTPUT_DIR}")
    print(f"\n📊 Validation dataset: {VAL_OUTPUT_DIR}")
    print(f"   - New crops (val only) for additional validation")
    print("\n🚀 Next step: Train final Stage 2 model")
    print("   python train_final_stage2.py")
    print("="*80)


if __name__ == "__main__":
    main()
