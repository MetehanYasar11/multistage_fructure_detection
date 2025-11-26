# Vision Transformer Mimarileri - Araştırma Özeti

## SORUN ANALİZİ
- **2 kez denedik, 2 kez F1 = 0.0000**
- Model hep aynı class'ı predict ediyor
- Custom MIL + Transformer yaklaşımı çalışmıyor
- Weakly-supervised localization başarısız

## MODERN VİSİON TRANSFORMER MİMARİLERİ

### 1. **ViT (Vision Transformer)** - Google Research
- **Paper**: "An Image is Worth 16x16 Words" (2020)
- **Mimari**: Pure transformer, patch embedding
- **장점**: Simple, scalable, SOTA on ImageNet
- **Kullanım**: `timm.create_model('vit_base_patch16_224')`

### 2. **Swin Transformer** - Microsoft
- **Paper**: "Swin Transformer: Hierarchical Vision Transformer" (2021)
- **Mimari**: Shifted windows, hierarchical structure
- **장점**: Efficient, multi-scale features
- **SOTA**: Object detection, segmentation
- **Kullanım**: `timm.create_model('swin_base_patch4_window7_224')`

### 3. **BEiT (BERT Pre-Training of Image Transformers)** - Microsoft
- **Paper**: "BEiT: BERT Pre-Training of Image Transformers" (2021)
- **Mimari**: Masked image modeling
- **장점**: Self-supervised pretraining

### 4. **DeiT (Data-efficient Image Transformers)** - Facebook
- **Paper**: "Training data-efficient image transformers" (2021)
- **Mimari**: Distillation token
- **장점**: Works with smaller datasets
- **Kullanım**: `timm.create_model('deit_base_patch16_224')`

### 5. **ConvNext** - Facebook AI
- **Paper**: "A ConvNet for the 2020s" (2022)
- **Mimari**: Modern CNN with transformer insights
- **장점**: Better than ViT on many tasks
- **Kullanım**: `timm.create_model('convnext_base')`

### 6. **MaxViT** - Google Research
- **Paper**: "MaxViT: Multi-Axis Vision Transformer" (2022)
- **Mimari**: Multi-axis self-attention
- **장점**: Efficient global + local attention

### 7. **SAM (Segment Anything Model)** - Meta
- **Paper**: "Segment Anything" (2023)
- **Mimari**: ViT encoder + mask decoder
- **장점**: Zero-shot segmentation
- **İDEAL**: Fracture localization!

### 8. **EVA (Exploring the Limits of Masked Visual Representation Learning)** - BAAI
- **Paper**: EVA (2022)
- **Mimari**: Large-scale masked autoencoder
- **장점**: 1B parameters, SOTA

## ÖNERİLER - DİŞ KIRIKLARI İÇİN

### **En İyi Seçenekler:**

#### **1. Swin Transformer + Classification Head** ⭐ **EN İYİ**
```python
# Swin-B pretrained on ImageNet-21k
model = timm.create_model(
    'swin_base_patch4_window7_224',
    pretrained=True,
    num_classes=2,  # Binary: Fractured vs Healthy
    img_size=224
)
```
**장점:**
- Hierarchical features (multi-scale)
- Shifted windows → efficient
- Medical imaging'de kanıtlanmış
- Fine-tuning kolay

#### **2. ConvNeXt + Class Attention** ⭐ **BASIT AMA GÜÇLÜ**
```python
model = timm.create_model(
    'convnext_base',
    pretrained=True,
    num_classes=2
)
```
**장점:**
- ViT kadar güçlü ama CNN efficiency
- Better generalization
- Less data needed

#### **3. DeiT + Attention Rollout** ⭐ **LOCALİZATION İÇİN**
```python
model = timm.create_model(
    'deit_base_distilled_patch16_224',
    pretrained=True,
    num_classes=2
)
```
**장점:**
- Attention rollout → fracture location
- Distillation token → better accuracy
- Built-in visualization

### **LOCALIZATION İÇİN ÖZEL YAKLAŞIMLAR:**

#### **A. Attention Rollout** (ViT/DeiT)
- Transformer attention maps
- No additional training
- Visualize which patches model focuses on

#### **B. Grad-CAM for ViT**
- Gradient-based visualization
- Works with any ViT
- Shows important regions

#### **C. SAM (Segment Anything Model)**
- Use pretrained SAM encoder
- Add classification head
- Get segmentation for free

#### **D. YOLO + Transformer Backbone**
- YOLOv8 with Swin backbone
- Direct fracture detection
- Bounding box output

## RECOMMENDED IMPLEMENTATION

### **Yaklaşım 1: Swin Transformer (Basit)**
```python
import timm
import torch.nn as nn

class FractureClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Swin-B pretrained
        self.backbone = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=True,
            num_classes=0,  # Remove head
            global_pool=''   # Remove global pool
        )
        
        # Get feature dimension
        num_features = self.backbone.num_features
        
        # Simple classification head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)  # Binary
        )
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)  # (B, L, C)
        
        # Classify
        logits = self.head(features.transpose(1, 2))
        
        return logits
```

### **Yaklaşım 2: DeiT + Attention Visualization**
```python
import timm

class FractureDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            'deit_base_distilled_patch16_224',
            pretrained=True,
            num_classes=2
        )
    
    def forward(self, x):
        return self.model(x)
    
    def get_attention_maps(self, x):
        # Get attention from all layers
        attentions = []
        def hook_fn(module, input, output):
            attentions.append(output)
        
        # Register hooks on attention layers
        for block in self.model.blocks:
            block.attn.register_forward_hook(hook_fn)
        
        _ = self.model(x)
        return attentions
```

### **Yaklaşım 3: SAM-based (SOTA Localization)**
```python
from segment_anything import sam_model_registry, SamPredictor

class SAMFractureDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # Load SAM encoder (ViT-H)
        sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
        self.encoder = sam.image_encoder
        
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        # SAM encoding
        features = self.encoder(x)  # (B, 1280, H/16, W/16)
        
        # Classify
        logits = self.classifier(features)
        
        return logits, features  # Return features for visualization
```

## KARŞILAŞTIRMA

| Model | Params | Speed | Accuracy | Localization | Ease |
|-------|--------|-------|----------|--------------|------|
| **Swin-B** | 88M | Fast | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **ConvNeXt** | 89M | Fastest | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **DeiT** | 86M | Medium | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **SAM** | 636M | Slow | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **ViT** | 86M | Medium | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## ÖNERİM

**1. Swin Transformer ile başla** (En balanced)
- Pretrained weights
- Fine-tune son 2 layer
- AdamW, lr=1e-4, warmup
- Focal loss for imbalance
- Gradient-based CAM for visualization

**2. Başarısız olursa → ConvNeXt**
- Daha simple
- Daha az overfitting

**3. Localization önemliyse → DeiT + Attention Rollout**
- Built-in attention maps
- No extra training

**Mevcut custom MIL yaklaşımını BİRAK!**
- 2 kez denedik, çalışmıyor
- Proven mimarileri kullan
- timm kütüphanesi → kolay implementation

Şimdi Swin Transformer implementasyonunu yapayım mı?
