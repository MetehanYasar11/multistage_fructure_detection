# 🎓 Thesis Presentation - Command Reference

## Quick Navigation
- [Model Evaluation](#model-evaluation)
- [Visualization Scripts](#visualization-scripts)
- [Training Scripts](#training-scripts)
- [Pipeline Testing](#pipeline-testing)
- [Best Results](#best-results)

---

## 🔬 Model Evaluation

### Stage 1: RCT Detection (YOLOv11x)
```bash
# Training results location
runs/detect/training_054345/

# Key metrics (500 epochs)
# - Precision: 81.05%
# - Recall: 75.77%
# - mAP50: 79.06%
# - mAP50-95: 59.25%

# Visualizations
outputs/FINAL_repo_visualizations/stage1_confusion_matrix_normalized.png
outputs/FINAL_repo_visualizations/stage1_results.png
```

### Stage 2: Fracture Classification (ViT-Small)
```bash
# Primary validation (184 crops with GT)
python evaluate_stage2_gt.py

# Expected output:
# - Accuracy: 84.78%
# - Precision: 89.19%
# - Recall: 88.71%
# - F1 Score: 88.95%

# Results location
runs/FINAL_vit_classifier/stage2_gt_evaluation/
outputs/FINAL_risk_zones_vit/stage2_gt_evaluation/
```

### 20-Image Professor Test
```bash
# Evaluate on 20 expert-labeled images
python evaluate_20_test_images.py

# Expected results:
# conf ≥ 0.5: 88.24% accuracy, 88.24% recall
# conf ≥ 0.3: 94.44% accuracy, 100% recall

# Results location
outputs/20_test_images_crop_level_evaluation.json
```

---

## 🎨 Visualization Scripts

### Risk Zone Visualization (PRIMARY - ViT-based)
```bash
# Generate risk zones for test images
python visualize_risk_zones_vit.py

# Outputs: Color-coded risk maps
# - GREEN: Confidence 0.0-0.3 (healthy)
# - YELLOW: Confidence 0.3-0.7 (uncertain)
# - RED: Confidence 0.7-1.0 (fractured)

# Output location
outputs/FINAL_risk_zones_vit/
```

### Improved Risk Zones v2 (BEST QUALITY)
```bash
# High-quality risk zone visualizations
python visualize_improved_risk_zones_v2.py

# Features:
# - Multi-scale crops (bounding box expansion 2.2x)
# - Crop-level predictions
# - Color-coded confidence zones

# Output location
outputs/FINAL_improved_risk_zones_v2/

# Best examples for presentation
outputs/FINAL_improved_risk_zones_v2/0039_risk_zones.jpg
outputs/FINAL_improved_risk_zones_v2/0052_risk_zones.jpg
```

### Repository Statistics
```bash
# Generate repository overview stats
python generate_repo_stats.py

# Visualize stats
python visualize_repo_stats.py

# Outputs
outputs/FINAL_repo_visualizations/repo_statistics_overview.png
outputs/FINAL_repo_visualizations/research_timeline.png
outputs/FINAL_repo_visualizations/experiments_breakdown.png
```

---

## 🏋️ Training Scripts

### Stage 2 Classification Training (ViT-Small)
```bash
# Train on SR+CLAHE preprocessed auto-labeled crops
python train_vit_sr_clahe_auto.py

# Configuration:
# - Model: vit_small_patch16_224
# - Dataset: ~1600 auto-labeled crops
# - Preprocessing: Super-Resolution (4x) + CLAHE
# - Weighted loss for class imbalance

# Output location
runs/FINAL_vit_classifier/
```

### Data Preprocessing
```bash
# Create auto-labeled crops from pipeline detections
python create_auto_labeled_crops.py

# Apply SR+CLAHE preprocessing
python preprocess_auto_labeled_sr_clahe.py

# Outputs
auto_labeled_crops_sr_clahe/
```

---

## 🔧 Pipeline Testing

### Full Pipeline Evaluation (Image-Level)
```bash
# Evaluate on Dataset_2021 (50 images with image-level GT)
python evaluate_pipeline_crop_level.py

# Metrics reported:
# - Image-level accuracy
# - Crop-level performance
# - Confusion matrix

# Results location
runs/FINAL_full_pipeline_validation/
```

### Pipeline Optimization (Grid Search)
```bash
# Grid search over confidence thresholds and bbox counts
# 120 configurations tested

# Best configuration found:
# - Confidence threshold: 0.75
# - Min bbox count: 2
# - 8× specificity improvement (7.69% → 61.54%)

# Results location
runs/FINAL_pipeline_optimization/grid_search_results.json
runs/FINAL_pipeline_optimization/grid_search_heatmaps.png
runs/FINAL_pipeline_optimization/sensitivity_vs_specificity.png
```

---

## 📊 Best Results (For Presentation)

### Key Metrics Table
| Evaluation | Accuracy | Recall | Precision | Notes |
|-----------|----------|--------|-----------|-------|
| **Stage 1 (Detection)** | - | 75.77% | 81.05% | mAP50: 79.06% |
| **Stage 2 (Classification)** | 84.78% | 88.71% | 89.19% | 184 crops with GT |
| **20-Image Test (conf≥0.5)** | 88.24% | 88.24% | - | Expert-labeled |
| **20-Image Test (conf≥0.3)** | 94.44% | 100% | - | Perfect recall |
| **Pipeline (Optimized)** | 89.47% | 82.35% | - | Image-level |

### Best Visual Examples
```bash
# Best risk zone visualizations
outputs/FINAL_improved_risk_zones_v2/0039_risk_zones.jpg  # GREEN zones (healthy)
outputs/FINAL_improved_risk_zones_v2/0052_risk_zones.jpg  # Color-coded confidence

# Training progression
runs/FINAL_vit_classifier/training_history.png
runs/FINAL_vit_classifier/confusion_matrix.png

# Pipeline optimization proof
runs/FINAL_pipeline_optimization/sensitivity_vs_specificity.png  # 8× improvement!
```

### Key Figures for Slides
```bash
# Stage 1: Detection
outputs/stage1_confusion_matrix_normalized.png
outputs/stage1_results.png

# Stage 2: Classification  
runs/FINAL_vit_classifier/stage2_confusion_matrix_gt.png
runs/FINAL_vit_classifier/stage2_evaluation_summary.png

# Preprocessing
outputs/sr_comparison_visualization.png
outputs/sr_detailed_steps.png
outputs/combined_clahe_gabor.png  # Failed experiment

# Pipeline Optimization
runs/FINAL_pipeline_optimization/grid_search_heatmaps.png
runs/FINAL_pipeline_optimization/top_10_configurations.png

# Visual Results
outputs/FINAL_visual_evaluation/fractured_examples_0012.png
outputs/FINAL_visual_evaluation/fractured_examples_0035.png
outputs/FINAL_visual_evaluation/healthy_examples_0043.png
outputs/FINAL_visual_evaluation/healthy_examples_0047.png

# Risk Zones (BEST FOR PRESENTATION!)
outputs/FINAL_risk_zones_vit/0039_risk_zones.jpg
outputs/FINAL_risk_zones_vit/0052_risk_zones.jpg
```

---

## 📁 Repository Structure

```
dental_fracture_detection/
├── detectors/                  # Trained models
│   ├── RCTdetector_v11x_v2.pt  # Stage 1: YOLOv11x
│   └── best_vit_classifier.pth # Stage 2: ViT-Small
├── outputs/                    # Evaluation results
│   ├── FINAL_risk_zones_vit/   # Primary risk zone visualizations
│   ├── FINAL_improved_risk_zones_v2/  # Best quality risk zones
│   ├── FINAL_visual_evaluation/       # Visual prediction examples
│   └── FINAL_repo_visualizations/     # Repository stats
├── runs/                       # Training runs
│   ├── FINAL_vit_classifier/   # Stage 2 training
│   ├── FINAL_pipeline_optimization/   # Grid search results
│   └── FINAL_full_pipeline_validation/ # Pipeline evaluation
├── documentation/              # Project documentation
└── archive/                    # Old experiments and scripts
```

---

## 🚀 Quick Demo Commands (For Presentation)

### 1. Show Repository Stats
```bash
python visualize_repo_stats.py
# Opens: Repository statistics overview
```

### 2. Generate Risk Zones for Image
```bash
python visualize_risk_zones_vit.py
# Outputs: outputs/FINAL_risk_zones_vit/
```

### 3. Evaluate Stage 2 Performance
```bash
python evaluate_stage2_gt.py
# Shows: 84.78% accuracy on 184 crops
```

### 4. Show Best Visual Results
```bash
# Open in image viewer
outputs/FINAL_improved_risk_zones_v2/0039_risk_zones.jpg
outputs/FINAL_improved_risk_zones_v2/0052_risk_zones.jpg
```

---

## 📈 Key Achievements (For Thesis Defense)

1. **Two-Stage Pipeline**
   - Stage 1: YOLOv11x detector (79.06% mAP50)
   - Stage 2: ViT-Small classifier (84.78% accuracy)

2. **Data Generation Innovation**
   - Auto-labeling: 200× faster (40-60 hours → 15 minutes)
   - 1604 auto-labeled crops generated
   - Liang-Barsky clipping algorithm for crop extraction

3. **Preprocessing Optimization**
   - SR+CLAHE: +4.63% improvement
   - Systematic comparison of 8 preprocessing strategies

4. **Class Imbalance Solution**
   - Weighted loss: +49.8pp recall improvement
   - Balanced training without data duplication

5. **Pipeline Optimization**
   - Grid search: 120 configurations
   - 8× specificity improvement (7.69% → 61.54%)
   - Confidence threshold tuning

6. **Strong Validation**
   - Primary: 84.78% (184 crops with GT)
   - 20-image test: 94.44% accuracy, 100% recall (conf≥0.3)
   - Image-level: 89.47% accuracy (optimized pipeline)

---

## 📞 Contact & Repository
- **Author**: Metehan Yaşar
- **Repository**: github.com/MetehanYasar11/multistage_fructure_detection
- **Branch**: prototype
- **Date**: December 2025

---

## 💡 Tips for Presentation

1. **Start with visual impact**: Show risk zone examples first
2. **Emphasize pipeline design**: Two-stage cascade architecture
3. **Highlight innovations**: Auto-labeling (200× speedup!)
4. **Show optimization journey**: Preprocessing experiments, failed attempts
5. **End with strong metrics**: 94.44% accuracy, 100% recall on expert test
6. **Be ready to demo**: Risk zone generation live (< 3 seconds/image)

**Good luck with your defense! 🎓✨**
