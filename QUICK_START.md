# 🚀 QUICK START - Thesis Defense Demo

## ⚡ 5-Minute Setup

### 1. Activate Environment
```bash
conda activate dental-ai
```

### 2. Verify Models Exist
```bash
# Check Stage 1 detector
ls detectors/RCTdetector_v11x_v2.pt

# Check Stage 2 classifier
ls detectors/best_vit_classifier.pth
```

### 3. Test Scripts (Choose ONE to demo)

#### Option A: Risk Zone Visualization (RECOMMENDED! ⭐)
```bash
python visualize_risk_zones_vit.py
```
**Output:** `outputs/FINAL_risk_zones_vit/`  
**Time:** 2-3 seconds per image  
**Best for:** Live demo during defense

#### Option B: Stage 2 Evaluation
```bash
python evaluate_stage2_gt.py
```
**Expected:** 84.78% accuracy  
**Time:** 10-15 seconds  
**Best for:** Showing validation metrics

#### Option C: 20-Image Professor Test
```bash
python evaluate_20_test_images.py
```
**Expected:** 94.44% accuracy, 100% recall  
**Time:** 30-40 seconds  
**Best for:** Showing expert validation

---

## 🖼️ Show Best Results (No Computation Needed!)

### Best Risk Zone Examples
```bash
# Open in image viewer
start outputs/FINAL_improved_risk_zones_v2/0039_risk_zones.jpg
start outputs/FINAL_improved_risk_zones_v2/0052_risk_zones.jpg
```

### Training Results
```bash
start runs/FINAL_vit_classifier/training_history.png
start runs/FINAL_vit_classifier/confusion_matrix.png
```

### Pipeline Optimization (8× improvement!)
```bash
start runs/FINAL_pipeline_optimization/sensitivity_vs_specificity.png
```

---

## 📊 Key Metrics (Memorize!)

| Evaluation | Accuracy | Recall | Notes |
|-----------|----------|--------|-------|
| Stage 2 (184 crops) | **84.78%** | 88.71% | Primary validation |
| 20-image test (conf≥0.3) | **94.44%** | **100%** | Perfect recall! |
| Pipeline (optimized) | **89.47%** | 82.35% | Image-level |

---

## 🎯 Demo Flow (Practice!)

### 1. Opening (30 seconds)
Show best risk zones:
```bash
start outputs/FINAL_improved_risk_zones_v2/0039_risk_zones.jpg
```
**Say:** "Color-coded confidence maps - GREEN=healthy, RED=fractured"

### 2. Architecture (1 minute)
- Stage 1: YOLOv11x (79.06% mAP50)
- Stage 2: ViT-Small (84.78% accuracy)

### 3. Innovation (1 minute)
- Auto-labeling: **200× speedup** (40-60 hours → 15 minutes)
- Generated 1604 crops automatically

### 4. Results (1 minute)
- Primary: 84.78% (184 crops)
- Expert test: 94.44%, **100% recall**
- Pipeline: 8× specificity improvement

### 5. Live Demo (30 seconds) - IF TIME
```bash
python visualize_risk_zones_vit.py
```

---

## 🚨 Troubleshooting

### "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### "CUDA out of memory"
Edit script: Change `device = 'cuda'` → `device = 'cpu'`  
(Demo will be slower but works)

### "File not found"
Check you're in project root:
```bash
pwd  # Should end with: dental_fracture_detection
```

---

## 📁 File Locations (Quick Reference)

- **Models:** `detectors/`
- **Best Results:** `outputs/FINAL_*/`
- **Training Runs:** `runs/FINAL_*/`
- **Scripts:** Root directory
- **Documentation:** `documentation/`

---

## 📞 Day-of-Defense Checklist

- [ ] Conda environment activated
- [ ] Tested risk zone script (python visualize_risk_zones_vit.py)
- [ ] Best images open in viewer (0039, 0052)
- [ ] Key metrics memorized (84.78%, 94.44%, 100%)
- [ ] Backup USB with slides + figures

---

**You got this! 🎓💪**

For detailed commands: See `PRESENTATION_README.md`  
For defense strategy: See `DEFENSE_PREPARATION_GUIDE.md`
