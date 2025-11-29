"""
Evaluate YOLOv11n SR+CLAHE model in detail
Check accuracy, sensitivity, specificity, and per-class performance
"""
from ultralytics import YOLO
import json
from pathlib import Path
import numpy as np

def evaluate_model_detailed(model_path, data_dir):
    print("\n" + "="*70)
    print("🔍 Detailed Evaluation: YOLOv11n SR+CLAHE")
    print("="*70)
    
    # Load model
    print(f"\n📦 Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Validate on test set
    print(f"\n📊 Running validation on test set...")
    results = model.val(data=data_dir, split='test')
    
    # Print detailed metrics
    print(f"\n" + "="*70)
    print(f"📈 DETAILED RESULTS")
    print(f"="*70)
    
    print(f"\n🎯 Overall Metrics:")
    print(f"   Top-1 Accuracy: {results.top1:.2f}%")
    print(f"   Top-5 Accuracy: {results.top5:.2f}%")
    
    # Get confusion matrix if available
    if hasattr(results, 'confusion_matrix') and results.confusion_matrix is not None:
        cm = results.confusion_matrix.matrix
        print(f"\n📊 Confusion Matrix:")
        print(f"   {cm}")
        
        # Calculate per-class metrics
        # Assuming class 0 = fractured, class 1 = healthy
        if cm.shape[0] >= 2:
            tp_frac = cm[0, 0]  # True Positive for fractured
            fn_frac = cm[0, 1]  # False Negative for fractured
            fp_frac = cm[1, 0]  # False Positive for fractured
            tn_frac = cm[1, 1]  # True Negative for fractured
            
            tp_heal = cm[1, 1]  # True Positive for healthy
            fn_heal = cm[1, 0]  # False Negative for healthy
            
            # Sensitivity (Recall) for fractured class
            sensitivity_frac = tp_frac / (tp_frac + fn_frac) if (tp_frac + fn_frac) > 0 else 0
            
            # Specificity for fractured class
            specificity_frac = tn_frac / (tn_frac + fp_frac) if (tn_frac + fp_frac) > 0 else 0
            
            # Recall for healthy class
            recall_heal = tp_heal / (tp_heal + fn_heal) if (tp_heal + fn_heal) > 0 else 0
            
            print(f"\n🔬 Per-Class Metrics:")
            print(f"\n   Fractured:")
            print(f"      Sensitivity (Recall): {sensitivity_frac:.2%}")
            print(f"      Specificity: {specificity_frac:.2%}")
            print(f"      True Positives: {tp_frac:.0f}")
            print(f"      False Negatives: {fn_frac:.0f}")
            print(f"      False Positives: {fp_frac:.0f}")
            
            print(f"\n   Healthy:")
            print(f"      Recall: {recall_heal:.2%}")
            print(f"      True Positives: {tp_heal:.0f}")
            print(f"      False Negatives: {fn_heal:.0f}")
    
    # Get per-class accuracy if available
    if hasattr(results, 'class_result'):
        print(f"\n📊 Class Results:")
        for idx, class_name in enumerate(model.names.values()):
            print(f"   {class_name}: {results.class_result(idx)}")
    
    # Compare with baseline
    print(f"\n" + "="*70)
    print(f"📊 COMPARISON WITH BASELINE")
    print(f"="*70)
    print(f"🏆 CLAHE Baseline:    84.70% acc, 72.73% frac recall")
    print(f"⭐ SR+CLAHE (nano):   {results.top1:.2f}% acc")
    
    if results.top1 > 84.70:
        print(f"✅ IMPROVEMENT: +{results.top1 - 84.70:.2f}%")
    else:
        print(f"❌ DECREASE: {results.top1 - 84.70:.2f}%")
    
    print(f"="*70)
    
    # Save detailed results
    detailed_results = {
        'model': 'YOLOv11n',
        'preprocessing': 'SR+CLAHE',
        'test_accuracy': float(results.top1),
        'top5_accuracy': float(results.top5),
        'model_path': str(model_path)
    }
    
    if hasattr(results, 'confusion_matrix') and results.confusion_matrix is not None:
        detailed_results['sensitivity_fractured'] = float(sensitivity_frac)
        detailed_results['specificity_fractured'] = float(specificity_frac)
        detailed_results['recall_healthy'] = float(recall_heal)
        detailed_results['confusion_matrix'] = cm.tolist()
    
    output_file = "outputs/yolo11n_sr_clahe_detailed.json"
    with open(output_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved: {output_file}")
    
    return results

if __name__ == "__main__":
    model_path = "runs/sr_clahe_models/yolo11n_sr_clahe/weights/best.pt"
    data_dir = "manual_annotated_crops_sr_clahe"
    
    results = evaluate_model_detailed(model_path, data_dir)
    
    print("\n🎯 Done!")
