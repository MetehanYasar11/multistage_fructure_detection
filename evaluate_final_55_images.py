"""
FINAL EVALUATION: Test on 55 images (Dataset_2021: 50 + new_data: 5)
"""

import torch
import timm
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import json
from ultralytics import YOLO

# Configuration
CONFIG = {
    'stage1_detector': 'detectors/RCTdetector_v11x_v2.pt',
    'stage2_classifier': 'detectors/FINAL_vit_classifier.pth',
    'test_images_dataset2021': 'path/to/Dataset_2021/test',  # 50 images
    'test_images_newdata': 'new_data_test_images',  # 5 images
    'output_dir': 'runs/FINAL_evaluation_55_images',
    
    # Detection config
    'conf_threshold': 0.3,
    'bbox_scale': 2.2,
    
    # Classification config
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def evaluate_55_images():
    """Evaluate pipeline on 55 test images"""
    
    print("="*80)
    print("FINAL EVALUATION: 55 Test Images (50 + 5)")
    print("="*80)
    
    # Load models
    print("\n📦 Loading models...")
    detector = YOLO(CONFIG['stage1_detector'])
    
    classifier = timm.create_model('vit_small_patch16_224', num_classes=2)
    classifier.load_state_dict(torch.load(CONFIG['stage2_classifier']))
    classifier = classifier.to(CONFIG['device'])
    classifier.eval()
    
    # Get test images
    dataset2021_images = list(Path(CONFIG['test_images_dataset2021']).glob('*.jpg'))
    newdata_images = list(Path(CONFIG['test_images_newdata']).glob('*.jpg'))
    
    all_test_images = dataset2021_images + newdata_images
    
    print(f"\n📊 Test images:")
    print(f"   Dataset_2021: {len(dataset2021_images)}")
    print(f"   new_data: {len(newdata_images)}")
    print(f"   TOTAL: {len(all_test_images)}")
    
    # Run evaluation
    results = []
    
    for img_path in all_test_images:
        print(f"\n   Processing: {img_path.name}")
        
        # Stage 1: Detection
        img = cv2.imread(str(img_path))
        detections = detector.predict(source=img, conf=CONFIG['conf_threshold'], verbose=False)
        
        # Extract and classify crops
        fractured_count = 0
        
        for result in detections:
            boxes = result.boxes
            
            for box in boxes:
                # Extract crop (with bbox expansion)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = (x2 - x1) * CONFIG['bbox_scale']
                h = (y2 - y1) * CONFIG['bbox_scale']
                
                x1_new = max(0, int(cx - w/2))
                y1_new = max(0, int(cy - h/2))
                x2_new = min(img.shape[1], int(cx + w/2))
                y2_new = min(img.shape[0], int(cy + h/2))
                
                crop = img[y1_new:y2_new, x1_new:x2_new]
                
                # Preprocess crop (SR+CLAHE)
                crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                crop_sr = cv2.resize(crop_gray, (crop_gray.shape[1]*4, crop_gray.shape[0]*4), 
                                    interpolation=cv2.INTER_CUBIC)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
                crop_clahe = clahe.apply(crop_sr)
                
                # Convert to tensor
                crop_pil = Image.fromarray(crop_clahe).convert('RGB')
                crop_tensor = torch.from_numpy(np.array(crop_pil)).permute(2, 0, 1).float() / 255.0
                crop_tensor = torch.nn.functional.interpolate(
                    crop_tensor.unsqueeze(0), size=(224, 224), mode='bilinear'
                )
                crop_tensor = (crop_tensor - 0.5) / 0.5
                crop_tensor = crop_tensor.to(CONFIG['device'])
                
                # Stage 2: Classification
                with torch.no_grad():
                    output = classifier(crop_tensor)
                    prob = torch.softmax(output, dim=1)[0]
                    prediction = prob[1].item()  # Fractured probability
                
                if prediction > 0.5:
                    fractured_count += 1
        
        # Image-level prediction
        image_prediction = "fractured" if fractured_count > 0 else "healthy"
        
        results.append({
            'image': img_path.name,
            'source': 'Dataset_2021' if img_path in dataset2021_images else 'new_data',
            'prediction': image_prediction,
            'fractured_crops': fractured_count
        })
        
        print(f"      Prediction: {image_prediction} ({fractured_count} fractured crops)")
    
    # Save results
    Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)
    with open(f"{CONFIG['output_dir']}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Evaluation completed!")
    print(f"   Results saved: {CONFIG['output_dir']}/results.json")
    print(f"\n💡 Now add ground truth labels and calculate metrics!")

if __name__ == '__main__':
    evaluate_55_images()
