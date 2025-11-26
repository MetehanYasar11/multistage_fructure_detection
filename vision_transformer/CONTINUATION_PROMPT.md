# 🦷 DENTAL FRACTURED INSTRUMENT DETECTION - CONTINUATION PROMPT

**Date**: October 28, 2025  
**Project**: Master's Thesis - Deep Learning for Dental X-Ray Analysis  
**Hardware**: NVIDIA RTX 5070 Ti (Blackwell sm_120, 16GB GDDR7, 300W TGP)  
**Environment**: Python 3.12, PyTorch 2.10.0.dev20251027+cu128, CUDA 12.8

---

## 📋 PROJECT OVERVIEW

### **Goal**
Build a high-performance CNN-Transformer hybrid network for **binary classification** of panoramic dental X-rays to detect fractured endodontic instruments with >80% accuracy and Dice score >0.84.

### **Dataset Understanding** (CRITICAL)
- **Total**: 521 panoramic X-ray images (2824-2958 × 1435 pixels, ~2:1 aspect ratio)
- **Fractured Class (407 images - POSITIVE)**: 
  - Contains **broken endodontic instruments** inside root canals
  - Annotations: ~1 line/vector per image marking the **fractured instrument location**
  - Higher brightness variance (94.56 ± 20.57) - more challenging
  - Average file size: 1919.63 KB
  
- **Healthy Class (114 images - NEGATIVE)**:
  - **Hard negative examples** - NO fractured instruments
  - Annotations: ~4.37 lines/vector per image marking **normal anatomical structures** (reference points, canal tips)
  - Lower brightness variance (86.05 ± 11.94) - more consistent
  - Average file size: 2030.12 KB
  - **Purpose**: Reduce false positives by training on challenging negative cases

- **Class Imbalance**: 3.57:1 (Fractured:Healthy) - reflects real-world rarity
- **Annotation Format**: 
  ```
  x1 y1  ← Vector 1 start point
  x2 y2  ← Vector 1 end point
  x3 y3  ← Vector 2 start point
  x4 y4  ← Vector 2 end point
  ```
  Each 2 consecutive lines = 1 line segment/vector

---

## ✅ COMPLETED TASKS

### **Task 1: Environment Setup** ✅
- **Conda Environment**: `dental-ai` with Python 3.12
- **PyTorch**: 2.10.0.dev20251027+cu128 (nightly build for Blackwell sm_120 support)
- **GPU Verification**: 
  ```
  PyTorch: 2.10.0.dev20251027+cu128
  CUDA: True
  GPU: NVIDIA GeForce RTX 5070 Ti
  Compute: sm_120
  ```
- **Installed Packages**: numpy, pandas, matplotlib, seaborn, scikit-learn, opencv, pillow, tqdm, pyyaml
- **Pending Packages** (install before continuing):
  - `timm==0.9.2` (EfficientNet, Swin Transformer backbones)
  - `albumentations==1.4.0` (medical image augmentation)
  - `wandb==0.16.3` (experiment tracking)
  - `torchmetrics==1.3.1` (metrics)
  - `grad-cam==1.5.0` (interpretability)
  - `segmentation-models-pytorch==0.3.3` (U2-Net)
  - `ultralytics==8.1.24` (YOLOv8 reference)
  - `einops==0.7.0` (tensor operations)
  - `tensorboard==2.16.2` (logging)
  - `onnx==1.15.0`, `onnxruntime-gpu==1.17.0` (deployment)

### **Task 2: Exploratory Data Analysis (EDA)** ✅
- **Script**: `data/eda.py` (fixed to handle Turkish characters in Windows paths with PIL)
- **Key Findings**:
  - All images have consistent height (1435px), variable width (2824-2958px)
  - Brightness: Fractured class has 72% higher variance → CLAHE preprocessing critical
  - Annotation discrepancy confirms hard negative mining strategy
  - Class imbalance managed: recommended weights [1.0, 3.57] for [Fractured, Healthy]
- **Outputs** (`outputs/eda/`):
  - `class_distribution.png` - Class balance visualization
  - `dimensions_comparison.png` - Dimension statistics
  - `pixel_intensity.png` - Intensity distributions
  - `annotation_distribution.png` - Annotation analysis
  - `boxplot_comparison.png` - Statistical comparisons
  - `samples_fractured.png` - Sample fractured instrument images with line annotations
  - `samples_healthy.png` - Sample healthy (hard negative) images with anatomical markers
  - `dataset_summary.json` - Complete numerical summary

---

## 🎯 NEXT TASKS (PRIORITIZED)

### **Task 3: Data Preprocessing Pipeline** 🔄 IN PROGRESS

**Objective**: Create robust PyTorch Dataset with medical imaging best practices

**Files to Create**:
1. `data/dataset.py` - Main PyTorch Dataset class
2. `data/augmentation.py` - Albumentations transforms
3. `data/split.py` - Stratified train/val/test split

**Critical Implementation Details**:

#### **3.1 Dataset Class (`data/dataset.py`)**
```python
class DentalXrayDataset(torch.utils.data.Dataset):
    """
    Binary classification: Fractured (1) vs Healthy (0)
    
    Key Features:
    - PIL image loading (handles Turkish characters in Windows paths)
    - CLAHE preprocessing (clip_limit=2.0, tile_grid_size=(8,8))
    - Resize strategies: 512×512, 640×640, 768×768 (test all)
    - Normalization: ImageNet stats (for pretrained models)
    - Optional: Annotation-based ROI extraction (future enhancement)
    """
```

**Required Parameters**:
- `root_dir`: `Dataset_2021/Dataset_2021/Dataset`
- `split`: 'train' / 'val' / 'test'
- `transform`: Albumentations compose
- `image_size`: 640 (default, configurable)
- `use_clahe`: True (default)

#### **3.2 Augmentation Strategy (`data/augmentation.py`)**

**Training Augmentations** (Albumentations):
```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    # 1. CLAHE - CRITICAL for brightness normalization
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
    
    # 2. Geometric - Panoramic X-rays tolerate small rotations
    A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
    A.HorizontalFlip(p=0.3),  # Symmetric dental structures
    
    # 3. Elastic transforms - Dental anatomy realism
    A.ElasticTransform(alpha=50, sigma=5, alpha_affine=10, p=0.3),
    A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),
    
    # 4. Intensity - Handle brightness variance
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
    
    # 5. Resize and normalize
    A.Resize(height=640, width=640),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# Validation/Test: Only CLAHE + Resize + Normalize
val_transform = A.Compose([
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
    A.Resize(height=640, width=640),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
```

**Advanced Augmentations** (Apply in training loop):
- **Mixup** (alpha=0.3, prob=0.3): Mix two images for regularization
- **CutMix** (alpha=1.0, prob=0.3): Cut-paste regions between images
- **Test-Time Augmentation (TTA)**: Average predictions over [original, h-flip, ±5° rotation]

#### **3.3 Data Split (`data/split.py`)**
```python
from sklearn.model_selection import StratifiedKFold

# Train: 70%, Val: 15%, Test: 15%
# STRATIFIED: Maintain 3.57:1 ratio in all splits
# Save split indices to JSON for reproducibility
```

**Class Balance Strategy**:
1. **Weighted Random Sampler** (training):
   - Fractured: weight = 1.0
   - Healthy: weight = 3.57 (oversample hard negatives)
   
2. **Focal Loss** (see Task 6):
   - `alpha=0.25` (more weight to Fractured/positive class)
   - `gamma=2.0` (focus on hard examples)

---

### **Task 4: Baseline Model - EfficientNet-B0** 🚀

**Objective**: Establish performance baseline with lightweight model

**File**: `models/efficientnet_classifier.py`

**Architecture**:
```python
import timm
import torch.nn as nn

class EfficientNetClassifier(nn.Module):
    def __init__(self, model_name='efficientnet_b0', pretrained=True, num_classes=1, dropout=0.3):
        super().__init__()
        # Load pretrained ImageNet weights
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # Custom classification head
        in_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout/2),
            nn.Linear(512, num_classes)  # Binary: 1 output (sigmoid)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
```

**Training Configuration**:
- **Input Size**: 512×512 (baseline), 640×640 (improved)
- **Loss**: BCEWithLogitsLoss + Focal Loss (weighted combination)
- **Optimizer**: AdamW (lr=1e-4, weight_decay=0.01, betas=(0.9, 0.999))
- **Scheduler**: CosineAnnealingWarmRestarts (T_0=10, T_mult=2)
- **Mixed Precision (AMP)**: `torch.cuda.amp.autocast()` - 2x speedup on RTX 5070 Ti
- **Batch Size**: 16 (baseline), tune [8, 16, 32]
- **Epochs**: 100 (early stopping patience=15)

**Target Metrics**:
- Accuracy: >70%
- Precision: >75% (minimize false positives - critical for Healthy hard negatives)
- Recall: >70%
- F1 Score: >70%

---

### **Task 5: U2-Net Model (SOTA 2025 Benchmark)** 🏆

**Objective**: Match/exceed 2025 Diagnostics paper (Dice=0.849, F1=0.861)

**File**: `models/u2net_classifier.py`

**Architecture**:
```python
import segmentation_models_pytorch as smp

class U2NetClassifier(nn.Module):
    def __init__(self, encoder='resnet34', encoder_weights='imagenet', num_classes=1):
        super().__init__()
        # U2-Net architecture with classification head
        self.encoder = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
        )
        
        # Global pooling for classification
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)  # ResNet34 output: 512
    
    def forward(self, x):
        # Extract features from U-Net encoder
        features = self.encoder.encoder(x)[-1]  # Bottleneck features
        pooled = self.pool(features).flatten(1)
        return self.classifier(pooled)
```

**Training Configuration**:
- **Input Size**: 640×640 (optimal for U2-Net)
- **Loss**: Dice Loss + BCE (α=0.5 weighted combination)
- **Encoder**: ResNet34 (pretrained ImageNet)
- **Optimizer**: AdamW (lr=5e-4, weight_decay=0.01)
- **Batch Size**: 8 (memory intensive)
- **Class Weights**: [1.0, 3.57] applied to loss

**Target Metrics** (Match 2025 SOTA):
- **Dice Score**: >0.84
- **F1 Score**: >0.86
- **Accuracy**: >80%

---

### **Task 6: Training Loop & Loss Functions** 🔥

**File**: `training/train.py`

**Loss Functions** (`training/losses.py`):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification
    
    Key for this dataset:
    - alpha=0.25: More weight to Fractured (positive) class
    - gamma=2.0: Focus on hard examples (misclassified samples)
    
    Formula: FL = -α(1-p)^γ log(p)
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

class CombinedLoss(nn.Module):
    """
    BCE + Focal Loss weighted combination
    
    Strategy:
    - BCE: Baseline binary classification
    - Focal: Handle class imbalance and hard examples
    - Label Smoothing: Prevent overconfidence (0.1)
    """
    def __init__(self, alpha=0.25, gamma=2.0, bce_weight=0.5, focal_weight=0.5, label_smoothing=0.1):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        # Apply label smoothing: 0 → 0.1, 1 → 0.9
        targets_smooth = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        bce_loss = self.bce(inputs, targets_smooth)
        focal_loss = self.focal(inputs, targets)
        
        return self.bce_weight * bce_loss + self.focal_weight * focal_loss
```

**Training Loop Features**:
1. **Mixed Precision (AMP)**: 
   ```python
   scaler = torch.cuda.amp.GradScaler()
   with torch.cuda.amp.autocast():
       outputs = model(images)
       loss = criterion(outputs, labels)
   ```

2. **Gradient Clipping**: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`

3. **Early Stopping**: Monitor validation loss, patience=15 epochs

4. **Mixup/CutMix** (in training loop):
   ```python
   from timm.data.mixup import Mixup
   mixup_fn = Mixup(mixup_alpha=0.3, cutmix_alpha=1.0, prob=0.3, mode='batch', label_smoothing=0.1)
   ```

5. **WandB Logging**:
   ```python
   import wandb
   wandb.init(project='dental-fracture-detection', config={...})
   wandb.log({
       'train_loss': loss.item(),
       'train_acc': accuracy,
       'val_dice': dice_score,
       'learning_rate': optimizer.param_groups[0]['lr'],
   })
   # Log sample predictions as images
   wandb.log({"predictions": [wandb.Image(img, caption=f"Pred: {pred}, GT: {label}")]})
   ```

---

### **Task 7: Grid Search & Cross-Validation** 🔬

**Objective**: Find optimal hyperparameters with WandB Sweeps

**File**: `training/sweep_config.yaml`

```yaml
program: training/train.py
method: bayes  # Bayesian optimization
metric:
  name: val_dice
  goal: maximize
parameters:
  learning_rate:
    values: [1e-4, 5e-4, 1e-3]
  batch_size:
    values: [8, 16, 32]
  image_size:
    values: [512, 640, 768]
  dropout:
    values: [0.2, 0.3, 0.5]
  focal_alpha:
    min: 0.2
    max: 0.4
  focal_gamma:
    values: [1.5, 2.0, 2.5]
  augmentation_prob:
    min: 0.2
    max: 0.5
```

**5-Fold Stratified Cross-Validation**:
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"=== Fold {fold+1}/5 ===")
    # Train model on train_idx, validate on val_idx
    # Track metrics per fold
```

**Metrics to Track**:
- **Classification**: Accuracy, Precision, Recall, F1, AUC-ROC
- **Segmentation-style**: Dice Score, IoU (if using attention maps)
- **Class-specific**: Precision/Recall for Fractured and Healthy separately
- **Confusion Matrix**: Track False Positives (critical for hard negatives)

---

### **Task 8: Evaluation & Documentation** 📊

**Files**:
- `evaluation/evaluate.py` - Test set evaluation
- `evaluation/grad_cam.py` - Interpretability visualization
- `evaluation/metrics.py` - Custom metrics

**Evaluation Pipeline**:

1. **Test Set Performance**:
   ```python
   from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
   
   # Load best model checkpoint
   model.load_state_dict(torch.load('checkpoints/best_model.pth'))
   model.eval()
   
   # Test set predictions
   all_preds, all_labels, all_probs = [], [], []
   with torch.no_grad():
       for images, labels in test_loader:
           outputs = model(images.cuda())
           probs = torch.sigmoid(outputs).cpu().numpy()
           preds = (probs > 0.5).astype(int)
           all_preds.extend(preds)
           all_labels.extend(labels.numpy())
           all_probs.extend(probs)
   
   # Metrics
   print(classification_report(all_labels, all_preds, target_names=['Healthy', 'Fractured']))
   print(confusion_matrix(all_labels, all_preds))
   
   # ROC Curve - Find optimal threshold
   fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
   optimal_idx = np.argmax(tpr - fpr)
   optimal_threshold = thresholds[optimal_idx]
   print(f"Optimal threshold: {optimal_threshold:.3f}")
   ```

2. **Grad-CAM Visualization**:
   ```python
   from pytorch_grad_cam import GradCAM
   from pytorch_grad_cam.utils.image import show_cam_on_image
   
   # Visualize what model focuses on
   cam = GradCAM(model=model, target_layers=[model.backbone.blocks[-1]])
   grayscale_cam = cam(input_tensor=images, targets=None)
   
   # Overlay on original X-ray
   visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
   
   # Save examples: Correct predictions, False Positives, False Negatives
   ```

3. **Benchmark Comparison**:
   - Compare against U2-Net 2025 paper: Dice=0.849, F1=0.861
   - Compare against baseline EfficientNet-B0
   - Report improvement percentages

4. **Model Export (ONNX)**:
   ```python
   # Export for deployment
   dummy_input = torch.randn(1, 3, 640, 640).cuda()
   torch.onnx.export(
       model,
       dummy_input,
       "checkpoints/model.onnx",
       input_names=['input'],
       output_names=['output'],
       dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
   )
   
   # Test ONNX inference
   import onnxruntime as ort
   sess = ort.InferenceSession("checkpoints/model.onnx", providers=['CUDAExecutionProvider'])
   ```

5. **Final Documentation**:
   - `README.md`: Project overview, setup instructions, results
   - `RESULTS.md`: Detailed metrics, confusion matrices, visualizations
   - `TRAINING_LOG.md`: Hyperparameters, training curves, lessons learned

---

## 🎯 SUCCESS CRITERIA (TARGET METRICS)

**Primary Goals**:
- ✅ **Accuracy**: >80%
- ✅ **Dice Score**: >0.84 (match U2-Net 2025 SOTA)
- ✅ **F1 Score**: >0.80
- ✅ **Precision**: >85% (minimize false positives - critical!)
- ✅ **Recall**: >75% (detect most fractured cases)

**Secondary Goals**:
- AUC-ROC: >0.90
- Inference Speed: <50ms per image (RTX 5070 Ti)
- Model Size: <100MB (deployment-ready)

---

## 🔧 TECHNICAL CONSTRAINTS & OPTIMIZATIONS

### **RTX 5070 Ti Blackwell Optimization**:
1. **Mixed Precision (AMP)**: Mandatory - 2x speedup, 50% memory reduction
2. **Batch Size**: Max 32 for 640×640 images (16GB VRAM)
3. **Gradient Checkpointing**: If memory issues with U2-Net
4. **DataLoader**: `num_workers=8`, `pin_memory=True`, `persistent_workers=True`

### **Windows + Turkish Characters**:
- ⚠️ **ALWAYS use PIL for image loading** (not OpenCV imread)
- Path handling: `from PIL import Image; Image.open(path)`

### **Reproducibility**:
```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

## 📁 PROJECT STRUCTURE

```
c:\Users\maspe\OneDrive\Masaüstü\masterthesis\dental_fracture_detection/
│
├── data/
│   ├── __init__.py
│   ├── eda.py                    ✅ DONE (fixed annotation parsing)
│   ├── dataset.py                🔄 TODO (Task 3)
│   ├── augmentation.py           🔄 TODO (Task 3)
│   └── split.py                  🔄 TODO (Task 3)
│
├── models/
│   ├── __init__.py
│   ├── efficientnet_classifier.py  🔄 TODO (Task 4)
│   ├── u2net_classifier.py         🔄 TODO (Task 5)
│   └── ensemble.py                 ⏳ OPTIONAL (Task 8)
│
├── training/
│   ├── __init__.py
│   ├── train.py                  🔄 TODO (Task 6)
│   ├── losses.py                 🔄 TODO (Task 6)
│   └── sweep_config.yaml         🔄 TODO (Task 7)
│
├── evaluation/
│   ├── __init__.py
│   ├── evaluate.py               🔄 TODO (Task 8)
│   ├── grad_cam.py               🔄 TODO (Task 8)
│   └── metrics.py                🔄 TODO (Task 8)
│
├── utils/
│   ├── __init__.py
│   ├── config.py                 🔄 TODO (Helper)
│   └── visualization.py          🔄 TODO (Helper)
│
├── outputs/
│   └── eda/                      ✅ DONE
│       ├── class_distribution.png
│       ├── dimensions_comparison.png
│       ├── pixel_intensity.png
│       ├── annotation_distribution.png
│       ├── boxplot_comparison.png
│       ├── samples_fractured.png
│       ├── samples_healthy.png
│       └── dataset_summary.json
│
├── checkpoints/                  (model weights saved here)
├── notebooks/                    (Jupyter experiments)
│
├── config.yaml                   ✅ DONE (needs update for new strategy)
├── environment.yml               ✅ DONE
├── requirements.txt              ✅ DONE
├── CONTINUATION_PROMPT.md        ✅ THIS FILE
└── README.md                     🔄 TODO (Task 8)
```

---

## 🚀 IMMEDIATE NEXT STEPS (WHEN RESUMING)

### **Step 1: Install Remaining Packages**
```bash
conda activate dental-ai
pip install timm==0.9.2 albumentations==1.4.0 wandb==0.16.3 torchmetrics==1.3.1 grad-cam==1.5.0 segmentation-models-pytorch==0.3.3 ultralytics==8.1.24 einops==0.7.0 tensorboard==2.16.2 onnx==1.15.0 onnxruntime-gpu==1.17.0
```

### **Step 2: Update Configuration**
Edit `config.yaml` with new strategy:
```yaml
# Class understanding
classes:
  fractured:
    label: 1
    description: "Broken endodontic instruments (positive class)"
    weight: 1.0
  healthy:
    label: 0
    description: "Hard negative examples (no fractures)"
    weight: 3.57

# Loss function
loss:
  type: "combined"  # BCE + Focal
  focal_alpha: 0.25  # More weight to Fractured class
  focal_gamma: 2.0
  bce_weight: 0.5
  focal_weight: 0.5
  label_smoothing: 0.1

# Data paths
dataset:
  root_dir: "c:/Users/maspe/OneDrive/Masaüstü/masterthesis/Dataset_2021/Dataset_2021/Dataset"
  fractured_dir: "Fractured"
  healthy_dir: "Healthy"
```

### **Step 3: Create Dataset Class**
Start with `data/dataset.py` - implement PIL-based loading, CLAHE, and transforms.

### **Step 4: Verify Data Pipeline**
```python
# Test script
from data.dataset import DentalXrayDataset
from data.augmentation import get_train_transforms

dataset = DentalXrayDataset(
    root_dir="Dataset_2021/Dataset_2021/Dataset",
    split="train",
    transform=get_train_transforms(image_size=640),
    use_clahe=True
)

# Visualize samples
import matplotlib.pyplot as plt
for i in range(5):
    img, label = dataset[i]
    plt.subplot(1, 5, i+1)
    plt.imshow(img.permute(1,2,0).numpy())
    plt.title(f"Label: {label}")
plt.show()
```

### **Step 5: Train Baseline (EfficientNet-B0)**
```bash
python training/train.py --model efficientnet_b0 --epochs 100 --batch_size 16 --lr 1e-4 --image_size 512
```

---

## 📚 KEY RESEARCH REFERENCES

1. **U2-Net Benchmark (2025 Diagnostics)**:
   - Dice: 0.849, F1: 0.861 for separated instrument detection
   - ResNet34 encoder, 640×640 input
   
2. **Focal Loss (Lin et al., 2017)**:
   - Formula: FL(pt) = -αt(1-pt)^γ log(pt)
   - Handles class imbalance without resampling
   
3. **CLAHE for Medical Imaging**:
   - Adaptive histogram equalization
   - Critical for X-ray brightness normalization

4. **Mixup/CutMix Augmentation**:
   - Regularization for small datasets
   - Reduces overfitting on imbalanced classes

---

## ⚠️ CRITICAL REMINDERS

1. **Always use PIL** for image loading (not cv2.imread) - Turkish character issue
2. **Fractured = Positive class** (label=1), **Healthy = Hard negatives** (label=0)
3. **False Positives are expensive** - Optimize for Precision (>85%)
4. **CLAHE is mandatory** - 72% brightness variance in Fractured class
5. **Mixed Precision (AMP)** - Required for RTX 5070 Ti efficiency
6. **Stratified splits** - Maintain 3.57:1 ratio in train/val/test
7. **WandB logging** - Track all experiments for comparison

---

## 🎓 EXPECTED DELIVERABLES

1. **Working Models**:
   - EfficientNet-B0 baseline (>70% accuracy)
   - U2-Net SOTA (>80% accuracy, Dice>0.84)
   - ONNX exported model for deployment

2. **Documentation**:
   - README with setup and usage
   - RESULTS.md with metrics and visualizations
   - Training logs and hyperparameter analysis

3. **Code Quality**:
   - Clean, modular architecture
   - Type hints and docstrings
   - Reproducible (seed setting, config files)

4. **Visualizations**:
   - Confusion matrices
   - ROC curves
   - Grad-CAM attention maps
   - Training/validation curves

---

## 💬 CONTINUATION MESSAGE

**Context for AI Assistant**:
```
You are continuing a master's thesis project on dental X-ray analysis for fractured endodontic instrument detection. The environment is set up (PyTorch 2.10 nightly, RTX 5070 Ti Blackwell sm_120), and EDA is complete. 

The dataset contains 407 Fractured (positive) and 114 Healthy (hard negative) panoramic X-rays. Key insight: Healthy images are challenging negatives to reduce false positives, not random negatives. Class imbalance (3.57:1) is expected and should be handled with Focal Loss (alpha=0.25, gamma=2.0) and weighted sampling.

Next task: Implement data preprocessing pipeline (Task 3) with:
- PyTorch Dataset using PIL (not OpenCV - Turkish character path issue)
- CLAHE preprocessing (critical - 72% brightness variance)
- Albumentations transforms (rotation ±15°, elastic, brightness)
- Stratified 70/15/15 split maintaining class ratio

Target: >80% accuracy, Dice>0.84, Precision>85% (minimize false positives).

All project files are in: c:\Users\maspe\OneDrive\Masaüstü\masterthesis\dental_fracture_detection\

Please proceed with implementing data/dataset.py, data/augmentation.py, and data/split.py as detailed in this prompt.
```

---

**Project Location**: `c:\Users\maspe\OneDrive\Masaüstü\masterthesis\dental_fracture_detection\`  
**Dataset Location**: `c:\Users\maspe\OneDrive\Masaüstü\masterthesis\Dataset_2021\Dataset_2021\Dataset\`  
**Environment**: `conda activate dental-ai`

**Good luck! 🚀🦷**
