# 🦷 Dental Fractured Instrument Detection

**Deep Learning for Panoramic Dental X-Ray Analysis**

Master's Thesis Project - October 2025

---

## 📋 Quick Start

### Environment Setup
```bash
# Activate conda environment
conda activate dental-ai

# Install remaining packages (if not done)
pip install timm==0.9.2 albumentations==1.4.0 wandb==0.16.3 torchmetrics==1.3.1 grad-cam==1.5.0 segmentation-models-pytorch==0.3.3 ultralytics==8.1.24 einops==0.7.0 tensorboard==2.16.2 onnx==1.15.0 onnxruntime-gpu==1.17.0

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

### Dataset Structure
```
Dataset_2021/Dataset_2021/Dataset/
├── Fractured/  (407 images - Positive class)
│   ├── 0001.jpg
│   ├── 0001.txt  (annotation: line vectors)
│   └── ...
└── Healthy/    (114 images - Hard negatives)
    ├── 0001.jpg
    ├── 0001.txt
    └── ...
```

---

## 🎯 Project Goal

Binary classification to detect **fractured endodontic instruments** in panoramic dental X-rays.

**Target Metrics**:
- Accuracy: **>80%**
- Dice Score: **>0.84** (SOTA benchmark)
- Precision: **>85%** (minimize false alarms)

---

## ✅ Completed

- [x] Environment setup (PyTorch 2.10 nightly, RTX 5070 Ti sm_120)
- [x] EDA analysis (see `outputs/eda/`)
- [x] Annotation format parsing (2 lines = 1 vector)

## 🔄 In Progress

- [ ] Data preprocessing pipeline (Task 3)
- [ ] EfficientNet-B0 baseline (Task 4)
- [ ] U2-Net SOTA model (Task 5)

---

## 📊 Dataset Insights

| Metric | Fractured (Positive) | Healthy (Negative) |
|--------|---------------------|-------------------|
| **Count** | 407 | 114 |
| **Purpose** | Broken instruments | Hard negatives |
| **Annotations** | ~1 line (fracture location) | ~4 lines (anatomy) |
| **Brightness** | 94.56 ± 20.57 | 86.05 ± 11.94 |
| **File Size** | 1919 KB | 2030 KB |

**Class Imbalance**: 3.57:1 (expected - reflects real-world rarity)

---

## 🚀 Next Steps

**See `CONTINUATION_PROMPT.md` for detailed instructions.**

1. Implement data preprocessing (`data/dataset.py`)
2. Train EfficientNet-B0 baseline
3. Train U2-Net SOTA model
4. Hyperparameter tuning with WandB Sweeps
5. Final evaluation and documentation

---

## 🔧 Hardware

- **GPU**: NVIDIA RTX 5070 Ti (Blackwell sm_120)
- **VRAM**: 16GB GDDR7
- **TGP**: 300W
- **CUDA**: 12.8
- **PyTorch**: 2.10.0.dev20251027+cu128

---

## 📚 References

1. U2-Net 2025 Diagnostics (Dice: 0.849, F1: 0.861)
2. Focal Loss (Lin et al., 2017)
3. CLAHE for Medical Imaging
4. Mixup/CutMix Augmentation

---

**Project Path**: `c:\Users\maspe\OneDrive\Masaüstü\masterthesis\dental_fracture_detection\`  
**Environment**: `dental-ai` (Python 3.12)

---

**Last Updated**: October 28, 2025
