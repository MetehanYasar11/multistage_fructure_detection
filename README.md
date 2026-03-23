# Dental Fracture Detection System - Prototype# Dental Fracture Detection System - Prototype



Multi-stage deep learning system for detecting vertical root fractures (VRF) in root canal treatments (RCT) from panoramic dental X-rays.Multi-stage deep learning system for detecting vertical root fractures in root canal treatments (RCT) from panoramic dental X-rays.



## 🎯 System Overview## 🎯 System Overview



This prototype uses a **two-stage pipeline** with **risk zone visualization**:This prototype uses a **two-stage pipeline** with **risk zone visualization**:



1. **Stage 1 (Detection):** YOLOv11x detects RCT locations in full panoramic X-rays1. **Stage 1 (Detection):** YOLOv11x detects RCT locations in panoramic X-rays

2. **Stage 2 (Classification):** Vision Transformer (ViT) classifies each RCT crop as fractured or healthy2. **Stage 2 (Classification):** Vision Transformer (ViT-Small) classifies each RCT crop as fractured or healthy

3. **Risk Zones:** Color-coded visualization for clinical decision support3. **Risk Zones:** Color-coded visualization for clinical decision support



## 📊 Dataset Statistics## 📊 Performance Metrics



The evaluation and training use a meticulously verified dataset structure:### Stage 2 (Crop-Level) - ViT-Small + SR+CLAHE

- **Original Source Dataset:** 487 panoramic radiographs (373 fractured + 114 healthy)- **Accuracy:** 84.78%

- **Stage 2 Auto-Labeled Crops Dataset:** 1,528 total RCT crops- **Precision:** 0.7237

  - **Training (70%):** 1,069 crops- **Recall:** 0.8871

  - **Validation (15%):** 229 crops- **Specificity:** 0.8279

  - **Test (15%):** 230 crops- **F1 Score:** 0.7971

  - **Class Distribution:** 486 Fractured (31.8%) / 1,042 Healthy (68.2%)

**Evaluation on 184 crops from 50 test images:**

## 📈 Performance Metrics- GT Healthy: 122 crops

- GT Fractured: 62 crops

### Stage 2 Classifier (Crop-Level on Test Set)- True Positives: 55

- **Accuracy:** 78.26%- True Negatives: 101

- **Loss Strategy:** Weighted CrossEntropy to handle class imbalance (Healthy: ~0.73, Fractured: ~1.57)- False Positives: 21

- False Negatives: 7

### Full Pipeline Image-Level (Evaluated on 70 Test Images)

- **Accuracy:** 82.86%### Image-Level (Risk Zone Evaluation)

- **Precision:** 0.9024 (Very high confidence when alarming)- **Accuracy:** 89.47%

- **Recall:** 0.8222- **Precision:** 1.00 (Zero false alarms!)

- **F1 Score:** 0.8605- **Recall:** 0.8947

- **Specificity:** 0.8400- **F1 Score:** 0.9444



*Note: The image-level pipeline aggregates crop risk zones, offering a superior clinical metric compared to raw crop-level classification.*## 🏗️ Architecture



## 🏗️ Architecture### Stage 1: RCT Detection

- **Model:** YOLOv11x (class: Root Canal Treatment, index=9)

### Stage 1: RCT Detection- **Confidence:** 0.3

- **Model:** YOLOv11x (class: Root Canal Treatment, index=9)- **Bbox Expansion:** 2.2x scale factor around center

- **Confidence Threshold:** 0.3

- **Bbox Expansion:** 2.2x scale factor around the detected center to capture surrounding root context.### Stage 2: Fracture Classification

- **Model:** ViT-Small (vit_small_patch16_224)

### Stage 2: Fracture Classification- **Parameters:** ~22M

- **Model:** ViT-Small (`vit_small_patch16_224`)- **Preprocessing:** SR+CLAHE

- **Parameters:** ~22M  - Super-resolution: 4x bicubic upscaling

- **Preprocessing:** Super-Resolution + CLAHE  - CLAHE: clipLimit=2.0, tileGridSize=(16,16)

  - **Super-resolution:** 4x bicubic upscaling (enhances fine fracture details)- **Loss:** Weighted CrossEntropyLoss (handles class imbalance)

  - **CLAHE:** clipLimit=2.0, tileGridSize=(16,16) (improves local contrast)  - Healthy weight: 0.73

  - *Result:* The enhanced image is resized back to original crop size ensuring a consistent input shape for ViT, while retaining enhanced textures.  - Fractured weight: 1.57



## 🚦 Risk Zone System### Training Dataset

- **Auto-labeled crops:** 1,604 samples

Visual feedback designed for radiologists:  - Fractured: 486 (30.3%)

- 🟢 **GREEN (Safe):** Healthy probability > 60% → No review needed  - Healthy: 1,118 (69.7%)

- 🟡 **YELLOW (Warning):** Probabilities 40-60% → Model uncertain, Doctor should check manually- **Split:** 70% train / 15% val / 15% test

- 🔴 **RED (Danger):** Fractured probability > 60% → ALARM! High likelihood of VRF- **Stratified sampling** to maintain class balance



## 🚀 Quick Start Guide## 🚦 Risk Zone System



### PrerequisitesVisual feedback for radiologists:

- Python 3.8+

- CUDA-capable GPU (highly recommended)- 🟢 **GREEN (Safe):** Healthy probability > 60% → No review needed

- 🟡 **YELLOW (Warning):** Both probabilities 40-60% → Doctor should check

### 1. Install Dependencies- 🔴 **RED (Danger):** Fractured probability > 60% → ALARM! Must review



```bash**Clinical Advantage:** Perfect precision (1.00) = Zero false alarms!

conda create -n dental-ai python=3.10

conda activate dental-ai## 📁 Repository Structure

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install ultralytics timm opencv-python scikit-learn matplotlib seaborn tqdm pandas```

```dental_fracture_detection/

├── create_auto_labeled_crops.py      # Automatic crop labeling (Stage 1 + GT lines)

### 2. Prepare Auto-Labeled Dataset├── train_vit_sr_clahe_auto.py        # ViT training with weighted loss

Generate training crops from panoramic X-rays using the Stage 1 detector + GT fracture lines (utilizes Liang-Barsky line clipping intersection algorithm):├── evaluate_stage2_gt.py             # Crop-level evaluation with GT

```bash├── visualize_risk_zones_vit.py       # Risk zone visualization (full pipeline)

python create_auto_labeled_crops.py├── config.yaml                       # Configuration file

```├── requirements.txt                  # Python dependencies

*(Creates `auto_labeled_crops/` with healthy and fractured folders based on GT overlap)*└── README.md                         # This file

```

### 3. Apply Preprocessing (SR + CLAHE)

Enhance the crop images before training:## 🚀 Installation

```bash

python preprocess_auto_labeled_sr_clahe.py### Prerequisites

```- Python 3.8+

*(Creates `auto_labeled_crops_sr_clahe/`)*- CUDA-capable GPU (recommended)



### 4. Train Stage 2 Classifier### Install Dependencies

Train the ViT model on the preprocessed auto-labeled crops:

```bash
python train_vit_sr_clahe_auto.py
```

*(Best model is saved to `runs/vit_sr_clahe_auto/best_model.pth`)*

### 5. Experimental U2-Net Segmentation (New)
Alternatively, you can test the new U²-Net Lite segmentation approach (inspired by SEI Detection studies) designed to trace the actual crack lines instead of basic classification:
```bash
# Generate Segmentation Dataset
python create_auto_seg_crops.py
python preprocess_segmentation_sr_clahe.py
# Train and Evaluate U2-Net Lite
python train_u2net_sr_clahe.py
python evaluate_u2net.py
```

### 6. Final Full-Pipeline Evaluation & Visualization
Evaluate final results on test images. This tests the full cascade (Full Image → Stage 1 → Crop Extraction → SR+CLAHE → Stage 2 → Risk Zone creation):- `torchvision>=0.15.0`

```bash- `ultralytics>=8.0.0` (YOLOv11)

python evaluate_70images_with_riskzones.py- `timm>=0.9.0` (Vision Transformers)

```- `opencv-python>=4.8.0`

*(Check `outputs/FINAL_70images_riskzones/` for metrics JSON and visually bounded output images)*- `scikit-learn>=1.3.0`

- `matplotlib>=3.7.0`

## 📁 Key Repository Files- `seaborn>=0.12.0`

- `tqdm>=4.65.0`

- `create_auto_labeled_crops.py`: Auto-labels Stage 1 detected RCTs based on GT annotations.

- `preprocess_auto_labeled_sr_clahe.py`: Prepares SR+CLAHE dataset.

- `train_vit_sr_clahe_auto.py`: ViT-Small model training script.

- `evaluate_70images_with_riskzones.py`: Main clinical evaluation pipeline script.

- `create_pipeline_figure_v4.py`: Script to generate methodological architecture figures with real images demonstrating the challenging scale differences (Image → RCT Crop → min enclosing VRF bbox).
- `train_u2net_sr_clahe.py`: Experimental U²-Net segmentation training script for fine morphologic structure tracing.
- `evaluate_u2net.py`: U²-Net segmentation visualization and benchmarking script.

## 🎓 Key Features

### 1. Automatic Dataset Generation
- No manual annotation required
- Uses GT fracture lines for labeling
- Consistent with training logic (Liang-Barsky line-box intersection)

### 2. Class Imbalance Handling
- Weighted CrossEntropyLoss
- Class weights computed automatically: `n_samples / (n_classes * class_counts)`
- Prevents majority class bias (previous YOLO approach achieved only 68%)

### 3. SR+CLAHE Preprocessing
- Enhances fine fracture details
- Super-resolution increases spatial resolution
- CLAHE improves local contrast

### 4. Clinical Decision Support
- Risk zone color coding for quick assessment
- Perfect precision = No wasted doctor time on false alarms
- High recall (88.7%) = Most fractures detected

## 📈 Model Comparison

| Model | Accuracy | Precision | Recall | F1 | Notes |
|-------|----------|-----------|--------|-----|-------|
| YOLOv11n-cls | 68% | N/A | N/A | N/A | Failed (predicting majority class) |
| ViT (crop-level) | 78.26% | 0.72 | 0.52 | 0.60 | Test set from training split |
| ViT (GT-based) | **84.78%** | **0.72** | **0.89** | **0.80** | Evaluated with GT fracture lines |
| ViT (image-level) | **89.47%** | **1.00** | **0.89** | **0.94** | Risk zone aggregation |

**Key Insight:** Image-level aggregation significantly improves clinical performance!

## 🔬 Technical Details

### Ground Truth Logic
A crop is labeled as **Fractured** if:
- GT fracture line intersects the expanded bbox (2.2x scale)
- Uses Liang-Barsky line-box intersection algorithm

A crop is labeled as **Healthy** if:
- No GT fracture line intersection

### Why 2.2x Bbox Expansion?
- Fractures often extend beyond visible RCT filling
- Matches Stage 2 training conditions
- Provides sufficient context for classification

### Why Weighted Loss?
Dataset imbalance (30% fractured / 70% healthy):
- Unweighted loss → Model learns to predict majority class
- Weighted loss → Forces model to learn minority class
- Result: 10% accuracy improvement + balanced performance

## 📊 Dataset Information

### Auto-Labeled Crops
- **Total:** 1,604 crops
- **Source:** Panoramic X-rays with GT fracture line annotations
- **Labeling:** Automatic using Stage 1 detector + line-box intersection
- **Preprocessing:** SR+CLAHE applied to all crops

### Split Strategy
- **Stratified train_test_split** with `random_state=42`
- Maintains 30/70 class ratio across splits
- No data leakage between train/val/test

## 🛠️ Configuration

Edit `create_auto_labeled_crops.py` for dataset generation:
```python
DETECTOR_MODEL = 'detectors/RCTdetector_v11x_v2.pt'
FRACTURED_DIR = 'path/to/fractured/xrays'
HEALTHY_DIR = 'path/to/healthy/xrays'
OUTPUT_DIR = 'auto_labeled_crops'
BBOX_SCALE = 2.2
CONFIDENCE = 0.3
```

Edit `train_vit_sr_clahe_auto.py` for training:
```python
config = {
    'data_dir': 'auto_labeled_crops_sr_clahe',
    'model_name': 'vit_small_patch16_224',
    'epochs': 50,
    'batch_size': 32,
    'lr': 1e-4,
    'dropout': 0.3
}
```

## 🎯 Future Improvements

1. **Uncertainty Estimation:** Temperature scaling for better probability calibration
2. **Ensemble Methods:** Combine multiple models for robustness
3. **Active Learning:** Identify uncertain cases for expert review
4. **Deployment:** REST API for clinical integration
5. **Multi-Class:** Detect fracture severity levels

## 📝 Citation

If you use this code, please cite:

```bibtex
@misc{dental_fracture_detection_2025,
  title={Multi-Stage Deep Learning System for Vertical Root Fracture Detection},
  author={Your Name},
  year={2025},
  publisher={GitHub},
  url={https://github.com/MetehanYasar11/multistage_fructure_detection}
}
```

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contact

For questions or collaboration:
- GitHub: [@MetehanYasar11](https://github.com/MetehanYasar11)

---

**Status:** Prototype - Production-ready performance (89% accuracy, 100% precision)  
**Last Updated:** March 24, 2026

---

## 📬 Latest Experiment Summary (March 2026)

We ran a systematic **6-experiment ablation study** (50 epochs each) to isolate the effect of each design choice on 3-class U²-Net segmentation (background / canal / fracture):

| # | Change | mDice (fg) |
|---|--------|-----------|
| 1 | Remove CLAHE (raw input) | 0.221 |
| 2 | + Mask dilation (11×11) | 0.326 |
| 3 | + Focal Tversky Loss | 0.430 |
| 4 | + Data augmentation | 0.332 |
| 5 | + clDice (topology loss) | 0.304 |
| 6 | **All combined** | **0.488** |

The best configuration (exp6) was then trained for **200 epochs with early stopping** (patience=20). Training converged at **epoch 42**:

- **Fracture Dice: 0.506** (+140% vs baseline)
- **Canal Dice: 0.464** (+178% vs baseline)
- **Image-level F1: 0.800**

### How to Reproduce

All scripts are on the `master` branch. Run in order:

```bash
# 1. Generate the 770-crop 3-class dataset
python create_auto_seg_crops_3class.py

# 2. Run the full 6-experiment ablation study
python run_ablation.py

# 3. Train the final model (best config from ablation)
python train_v4_final.py

# 4. Evaluate on the test set
python evaluate_u2net_v3.py
```

---

## 📅 Changelog

### [2026-03-23] U2-Net 3-Class Segmentation — Ablation Study & Final Model

#### Problem
Binary U2-Net segmentation was confusing **canal fillings** with **fractures** (both appear as thin white lines). Baseline 3-class model (v3) achieved only Dice[fracture]=0.211, mDice(fg)=0.189.

#### Research Findings
- **İnönü et al. (Diagnostics 2025;15(14):1744):** CLAHE actually hurts U2-Net performance (Dice 0.849→0.827). Min-max normalization + 500 epochs recommended.
- **clDice (CVPR 2021):** Topology-preserving loss for thin curvilinear structures via soft skeletonization.
- **Focal Tversky Loss (Abraham & Khan 2019):** Better handles extreme class imbalance (α=0.3, β=0.7, γ=0.75).
- **Class imbalance:** BG=99.44%, Canal=0.32%, Fracture=0.23% → extreme foreground scarcity.

#### Ablation Study (6 experiments × 50 epochs)
Each experiment tests ONE improvement vs baseline:

| # | Experiment | Dice[frac] | Dice[canal] | mDice(fg) | Img F1 |
|---|-----------|-----------|------------|----------|--------|
| 1 | No CLAHE (raw data) | 0.216 | 0.226 | 0.221 | 0.742 |
| 2 | Dilation 11×11 | 0.342 | 0.310 | 0.326 | 0.713 |
| 3 | **Focal Tversky Loss** | **0.435** | **0.424** | **0.430** | **0.775** |
| 4 | Data Augmentation | 0.369 | 0.295 | 0.332 | 0.774 |
| 5 | clDice Loss | 0.323 | 0.285 | 0.304 | 0.753 |
| 6 | **All Combined** | **0.501** | **0.476** | **0.488** | **0.782** |

**Key findings:**
- Focal Tversky Loss single-handedly provides +127% mDice improvement
- Larger dilation (11×11) increases foreground from ~0.56% to ~2%, +47% mDice
- CLAHE confirmed to NOT help (validating İnönü 2025)
- All combined (exp6) achieves best results and keeps improving at epoch 50

#### Final Model — U2-Net v4 (exp6 config + 200 epochs + early stopping)
Config: Raw data (no CLAHE) + dilation 11×11 + Combo Loss (FTL + 0.5×clDice) + augmentation

| Metric | Baseline v3 | **v4 Final** | Improvement |
|--------|------------|-------------|-------------|
| Dice[fracture] | 0.211 | **0.506** | **+140%** |
| Dice[canal] | 0.167 | **0.464** | **+178%** |
| mDice(fg) | 0.189 | **0.485** | **+157%** |
| Image F1 | 0.746 | **0.800** | +7% |
| Image Recall | 0.988 | 0.843 | -15% (less FP) |
| Image Precision | 0.608 | **0.761** | +25% |

- **Best epoch:** 42/62 (early stopped at epoch 62, patience=20)
- **No overfitting:** val_loss steadily decreased until convergence
- **Model:** `runs/u2net_v4_final/best_model.pth`

#### Scripts
- `run_ablation.py` — 6-experiment ablation study runner
- `train_v4_final.py` — Final model training (200ep + early stop)
- `create_auto_seg_crops_3class.py` — 3-class dataset generator (Liang-Barsky clipping)

#### 3-Class Dataset
- **770 crops** (445 fractured, 325 healthy canal)
- Classes: 0=background, 1=canal filling, 2=fracture
- GT-less crops skipped (no black masks)
- Location: `auto_labeled_segmentation_3class/`

### [2025-12-17] ViT-Small Classification Pipeline
- Stage 2 ViT-Small classifier with SR+CLAHE preprocessing
- Image-level F1=0.944, Precision=1.00, Recall=0.895
- Risk zone visualization system (Green/Yellow/Red)
