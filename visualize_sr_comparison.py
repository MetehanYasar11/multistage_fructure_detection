"""
Visualize SR vs No-SR predictions on sample crops
Shows original, SR-enhanced version, and predictions with confidence scores
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import random
from ultralytics import YOLO

def apply_clahe(img, clip_limit=2.0, tile_size=16):
    """Apply CLAHE preprocessing"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(img)

def apply_super_resolution_bicubic(img, scale=4):
    """Apply bicubic 4x upscaling"""
    h, w = img.shape[:2]
    enhanced = cv2.resize(img, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
    return enhanced

def predict_crop(model, crop, use_sr=False):
    """Predict crop class with optional SR"""
    # Apply SR if requested
    if use_sr:
        crop = apply_super_resolution_bicubic(crop, scale=4)
    
    # Apply CLAHE
    crop_clahe = apply_clahe(crop)
    
    # Convert to BGR for model
    if len(crop_clahe.shape) == 2:
        crop_clahe = cv2.cvtColor(crop_clahe, cv2.COLOR_GRAY2BGR)
    
    # Resize to model input
    crop_resized = cv2.resize(crop_clahe, (640, 640))
    
    # Predict
    results = model.predict(crop_resized, verbose=False)
    
    # Get prediction
    probs = results[0].probs
    pred_idx = probs.top1
    confidence = probs.top1conf.item()
    pred_class = model.names[pred_idx]
    
    return pred_class, confidence, crop_clahe

def create_comparison_visualization(crops_dir, model_path, num_samples=6):
    """Create visualization showing all preprocessing steps: Original -> CLAHE -> SR -> SR+CLAHE"""
    print("\n" + "="*70)
    print("🎨 Creating Detailed SR Preprocessing Visualization")
    print("="*70)
    
    # Load model
    print(f"\n📦 Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Get crops
    crops_dir = Path(crops_dir)
    frac_imgs = list((crops_dir / 'fractured').glob("*"))
    heal_imgs = list((crops_dir / 'healthy').glob("*"))
    
    print(f"📊 Found {len(frac_imgs)} fractured, {len(heal_imgs)} healthy crops")
    
    # Sample crops (mix of both classes)
    random.seed(42)
    sample_frac = random.sample(frac_imgs, min(num_samples//2, len(frac_imgs)))
    sample_heal = random.sample(heal_imgs, min(num_samples//2, len(heal_imgs)))
    sample_crops = sample_frac + sample_heal
    random.shuffle(sample_crops)
    sample_crops = sample_crops[:num_samples]
    
    print(f"🎯 Visualizing {len(sample_crops)} sample crops")
    
    # Create figure with 6 columns: Original, CLAHE, SR, SR+CLAHE, Pred(CLAHE), Pred(SR+CLAHE)
    fig, axes = plt.subplots(num_samples, 6, figsize=(24, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, crop_path in enumerate(sample_crops):
        # Load image
        crop = cv2.imread(str(crop_path))
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # Get ground truth
        gt_label = "Fractured" if "fractured" in str(crop_path) else "Healthy"
        
        # Step 1: Original
        original_rgb = crop_rgb
        
        # Step 2: CLAHE only
        clahe_only = apply_clahe(crop)
        
        # Step 3: SR only (4x upscale then downscale for visualization)
        sr_only = apply_super_resolution_bicubic(crop_gray, scale=4)
        h_target, w_target = crop_gray.shape[:2]
        sr_only_display = cv2.resize(sr_only, (w_target, h_target))
        
        # Step 4: SR + CLAHE (apply SR first, then CLAHE)
        sr_then_clahe = apply_clahe(sr_only)
        sr_then_clahe_display = cv2.resize(sr_then_clahe, (w_target, h_target))
        
        # Get predictions
        pred_no_sr, conf_no_sr, _ = predict_crop(model, crop, use_sr=False)
        pred_sr, conf_sr, _ = predict_crop(model, crop, use_sr=True)
        
        # Determine correctness
        correct_no_sr = pred_no_sr == gt_label
        correct_sr = pred_sr == gt_label
        changed = pred_no_sr != pred_sr
        
        # Plot each step
        
        # Column 1: Original
        axes[idx, 0].imshow(original_rgb)
        axes[idx, 0].set_title(f"1. Original\nGT: {gt_label}", fontsize=10, fontweight='bold')
        axes[idx, 0].axis('off')
        
        # Column 2: CLAHE only
        axes[idx, 1].imshow(clahe_only, cmap='gray')
        axes[idx, 1].set_title(f"2. CLAHE\n(clip=2.0)", fontsize=10)
        axes[idx, 1].axis('off')
        
        # Column 3: SR only
        axes[idx, 2].imshow(sr_only_display, cmap='gray')
        axes[idx, 2].set_title(f"3. SR 4x\n(Bicubic)", fontsize=10)
        axes[idx, 2].axis('off')
        
        # Column 4: SR + CLAHE
        axes[idx, 3].imshow(sr_then_clahe_display, cmap='gray')
        axes[idx, 3].set_title(f"4. SR + CLAHE\n(Combined)", fontsize=10)
        axes[idx, 3].axis('off')
        
        # Column 5: Prediction without SR (CLAHE only)
        pred_color_no_sr = 'green' if correct_no_sr else 'red'
        axes[idx, 4].text(0.5, 0.5, f"{pred_no_sr}\n{conf_no_sr:.2%}", 
                          ha='center', va='center', fontsize=14,
                          color=pred_color_no_sr, fontweight='bold',
                          transform=axes[idx, 4].transAxes)
        axes[idx, 4].set_title(f"Pred: CLAHE", fontsize=10, fontweight='bold')
        axes[idx, 4].axis('off')
        axes[idx, 4].set_facecolor('white')
        
        # Column 6: Prediction with SR (SR+CLAHE)
        pred_color_sr = 'green' if correct_sr else 'red'
        axes[idx, 5].text(0.5, 0.5, f"{pred_sr}\n{conf_sr:.2%}", 
                         ha='center', va='center', fontsize=14,
                         color=pred_color_sr, fontweight='bold',
                         transform=axes[idx, 5].transAxes)
        title_text = f"Pred: SR+CLAHE"
        if changed:
            title_text += " ⚡"  # Mark changed predictions
        axes[idx, 5].set_title(title_text, fontsize=10, fontweight='bold')
        axes[idx, 5].axis('off')
        axes[idx, 5].set_facecolor('white')
        
        # Print details
        status = "✅ Correct" if correct_sr else "❌ Wrong"
        change_status = "🔄 Changed" if changed else "  Same"
        print(f"  {idx+1}. {crop_path.name[:30]:30s} | GT: {gt_label:10s} | "
              f"CLAHE: {pred_no_sr:10s} {conf_no_sr:.2%} | "
              f"SR+CLAHE: {pred_sr:10s} {conf_sr:.2%} | {change_status} | {status}")
    
    plt.tight_layout()
    
    # Save
    output_path = "outputs/sr_detailed_steps.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved: {output_path}")
    
    plt.show()
    
    print("🎯 Done!")

if __name__ == "__main__":
    # Configuration
    crops_dir = "manual_annotated_crops"
    model_path = "runs/preprocess_grid/clahe_2.0_16/weights/best.pt"
    
    # Create visualization
    create_comparison_visualization(
        crops_dir=crops_dir,
        model_path=model_path,
        num_samples=8  # Show 8 examples
    )
