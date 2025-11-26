# OBJECT DETECTION MİMARİLERİ - Diş Kırığı Tespiti

## SORUN: Classifier değil, DETECTOR lazım!
- **Hedef**: Fracture'ın yerini bounding box ile göster
- **Current**: Sadece "var/yok" classification
- **Gerekli**: X, Y, W, H koordinatları

## MODERN OBJECT DETECTION MİMARİLERİ

### 1. **YOLOv8** ⭐⭐⭐⭐⭐ **EN İYİ SEÇİM**
- **Framework**: Ultralytics YOLO
- **Speed**: REAL-TIME (RTX 5070 Ti'da 100+ FPS)
- **Accuracy**: SOTA object detection
- **Kullanım**: Çok kolay, 10 satır kod
- **Pretrained**: COCO dataset
- **Fine-tuning**: Custom dataset için super kolay

```python
from ultralytics import YOLO

# Pretrained model yükle
model = YOLO('yolov8x.pt')  # x = en büyük, en accurate

# Custom dataset ile train et
model.train(
    data='dental_fracture.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0
)

# Detect
results = model.predict('test_image.jpg')
boxes = results[0].boxes  # Bounding boxes
```

**장점:**
- ✅ Çok hızlı
- ✅ Kolay kullanım
- ✅ Built-in visualization
- ✅ Export ONNX, TensorRT
- ✅ Medical imaging'de kanıtlanmış

### 2. **Faster R-CNN** ⭐⭐⭐⭐
- **Framework**: Detectron2 (Facebook)
- **Accuracy**: Çok yüksek
- **Speed**: Yavaş (2-stage detector)
- **Kullanım**: Orta zorluk

```python
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Sadece fracture

trainer = DefaultTrainer(cfg)
trainer.train()
```

**장점:**
- ✅ Çok accurate
- ✅ Small object detection iyi
- ❌ Yavaş
- ❌ Setup zor

### 3. **DETR (DEtection TRansformer)** ⭐⭐⭐⭐
- **Paper**: Facebook AI (2020)
- **Mimari**: Pure transformer
- **Accuracy**: Yüksek
- **Speed**: Orta

```python
from transformers import DetrImageProcessor, DetrForObjectDetection

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Fine-tune for dental fractures
```

**장점:**
- ✅ No anchor boxes
- ✅ Set prediction
- ✅ Modern yaklaşım
- ❌ Training uzun

### 4. **RT-DETR** ⭐⭐⭐⭐⭐ **REAL-TIME TRANSFORMER**
- **Paper**: Baidu (2023)
- **Mimari**: DETR + real-time optimization
- **Speed**: YOLOv8 seviyesinde
- **Accuracy**: DETR seviyesinde

```python
from ultralytics import RTDETR

model = RTDETR('rtdetr-x.pt')
model.train(data='dental_fracture.yaml', epochs=100)
```

**장점:**
- ✅ Transformer장점 + YOLO hız
- ✅ Ultralytics entegrasyonu
- ✅ Kolay kullanım

### 5. **SAM (Segment Anything)** ⭐⭐⭐ **SEGMENTATION**
- **Use Case**: Eğer bounding box değil de mask istersen
- **Accuracy**: Çok yüksek
- **Speed**: Yavaş

### 6. **EfficientDet** ⭐⭐⭐
- **Paper**: Google Brain
- **Accuracy**: Yüksek
- **Efficiency**: Optimize edilmiş

## VERİ FORMATI GEREKSİNİMİ

Şu an dataset'inde **bounding box annotations** var mı?

### Mevcut Format:
```json
{
  "image_path": "Fractured/001.jpg",
  "label": 1,
  "bboxes": [...]  // VAR MI?
}
```

### Gerekli Format (YOLO):
```
# data/labels/train/001.txt
0 0.5 0.3 0.2 0.15  # class x_center y_center width height (normalized)
```

### Gerekli Format (COCO - Detectron2):
```json
{
  "images": [...],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": ...,
      "iscrowd": 0
    }
  ],
  "categories": [{"id": 1, "name": "fracture"}]
}
```

## ÖNERİLER

### **EĞER BBOX ANNOTATIONS VARSA:**

#### **YOLOv8 ile devam** (En kolay + En iyi)
```bash
# Install
pip install ultralytics

# Dataset hazırla
data/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
  dental_fracture.yaml
```

```yaml
# dental_fracture.yaml
path: /path/to/data
train: images/train
val: images/val
test: images/test

nc: 1  # number of classes
names: ['fracture']
```

```python
from ultralytics import YOLO

# Train
model = YOLO('yolov8x.pt')
results = model.train(
    data='dental_fracture.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    patience=20,
    save=True,
    plots=True
)

# Validate
metrics = model.val()

# Predict
results = model.predict('test_image.jpg', save=True, conf=0.25)
```

### **EĞER BBOX ANNOTATIONS YOKSA:**

#### **Seçenek 1: LabelImg ile annotate et**
```bash
pip install labelImg
labelImg
```
- Manuel olarak her resimde fracture'ı işaretle
- 340 train image → ~2-3 saat iş

#### **Seçenek 2: Weakly-supervised detection**
- **WSDDN** - Weakly Supervised Deep Detection Networks
- Sadece image-level label ile bbox öğrenme
- Daha az accurate

#### **Seçenek 3: Class Activation Map (CAM) → Pseudo-labels**
1. Baseline classifier'ı kullan
2. Grad-CAM ile heatmap oluştur
3. Heatmap'ten bounding box çıkar (thresholding)
4. Pseudo-labels ile detector train et

## RECOMMENDED WORKFLOW

### **1. Dataset kontrol**
```python
# Check if bboxes exist
import json
data = json.load(open('dataset.json'))
print("Has bboxes:", 'bbox' in data[0] or 'bboxes' in data[0])
```

### **2A. Eğer bbox varsa → YOLOv8**
```python
# Convert to YOLO format
# Train YOLOv8
# Evaluate
# Visualize detections
```

### **2B. Eğer bbox yoksa → Grad-CAM + Pseudo-labels**
```python
# 1. Baseline model ile Grad-CAM
# 2. Heatmap → bbox conversion
# 3. Manual verification (sample check)
# 4. YOLOv8 training
```

## KARŞILAŞTIRMA

| Model | mAP | Speed (FPS) | Ease | Best For |
|-------|-----|-------------|------|----------|
| **YOLOv8x** | ⭐⭐⭐⭐⭐ | 100+ | ⭐⭐⭐⭐⭐ | **GENEL** |
| **RT-DETR** | ⭐⭐⭐⭐⭐ | 80+ | ⭐⭐⭐⭐ | Transformer fans |
| **Faster R-CNN** | ⭐⭐⭐⭐⭐ | 10 | ⭐⭐⭐ | Accuracy >> Speed |
| **DETR** | ⭐⭐⭐⭐ | 30 | ⭐⭐⭐ | Research |

## SON ÖNERİ

**YOLOv8x kullan!**

1. **Basit**: 10 satır kod
2. **Hızlı**: Real-time detection
3. **Accurate**: SOTA performance
4. **Proven**: Medical imaging'de çok kullanılıyor
5. **Tools**: Built-in visualization, export, etc.

Şimdi:
1. Dataset'inde bbox var mı kontrol edelim
2. Varsa → YOLO format'a çevir
3. Yoksa → Grad-CAM ile pseudo-label oluştur
4. YOLOv8 training pipeline kur

Hangi yol? 🚀
