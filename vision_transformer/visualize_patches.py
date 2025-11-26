"""
Patch Visualization Script
==========================

Model'in hangi patch'lere odaklandığını görselleştirmek için script.
Her görüntü için 392 patch'in (14x28 grid) prediction'larını heat map olarak gösterir.

Kullanım:
    python visualize_patches.py --num_samples 10
    python visualize_patches.py --image_id "specific_image_name.jpg"
    python visualize_patches.py --error_analysis --num_samples 5
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import cv2
from typing import List, Tuple, Optional
import seaborn as sns

# Configurasyon
DATA_ROOT = Path(r"c:\Users\maspe\OneDrive\Masaüstü\masterthesis\dental_fracture_detection\data")
OUTPUT_DIR = Path(r"c:\Users\maspe\OneDrive\Masaüstü\masterthesis\dental_fracture_detection\outputs\patch_visualizations")
TEST_RESULTS_DIR = Path(r"c:\Users\maspe\OneDrive\Masaüstü\masterthesis\dental_fracture_detection\outputs\test_evaluation")

# Patch configurasyon
PATCH_SIZE = 100
IMAGE_SIZE = (1400, 2800)  # H, W
NUM_PATCHES_H = IMAGE_SIZE[0] // PATCH_SIZE  # 14
NUM_PATCHES_W = IMAGE_SIZE[1] // PATCH_SIZE  # 28


class PatchVisualizer:
    """Test set için patch attention haritalarını görselleştirir."""
    
    def __init__(self):
        """Visualizer'ı başlatır ve test sonuçlarını yükler."""
        self.output_dir = OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test sonuçlarını yükle
        print("📂 Test sonuçları yükleniyor...")
        self._load_test_results()
        
        # Colormap'i ayarla
        self.cmap = sns.color_palette("RdYlGn_r", as_cmap=True)  # Kırmızı=yüksek risk
        
    def _load_test_results(self):
        """Test evaluation sonuçlarını yükler."""
        # Predictions (JSON dict of lists)
        pred_path = TEST_RESULTS_DIR / "test_predictions.json"
        with open(pred_path, 'r') as f:
            pred_data = json.load(f)
        
        # Convert to list of dicts
        num_samples = len(pred_data['predictions'])
        self.predictions = []
        for i in range(num_samples):
            self.predictions.append({
                'prediction': pred_data['predictions'][i],
                'target': pred_data['targets'][i],
                'probability': pred_data['probabilities'][i],
                'image_path': pred_data['image_paths'][i]
            })
        
        # Patch predictions (74, 392, 1) -> squeeze and apply sigmoid
        patch_pred_path = TEST_RESULTS_DIR / "test_patch_predictions.npy"
        patch_preds = np.load(patch_pred_path)
        patch_preds = patch_preds.squeeze()  # Remove last dimension -> (74, 392)
        
        # CRITICAL FIX: Apply sigmoid to convert logits to probabilities
        self.patch_predictions = 1 / (1 + np.exp(-patch_preds))  # Sigmoid
        
        print(f"✅ {len(self.predictions)} test görüntüsü yüklendi")
        print(f"✅ Patch predictions shape: {self.patch_predictions.shape}")
        print(f"✅ Patch prob range: [{self.patch_predictions.min():.3f}, {self.patch_predictions.max():.3f}]")
        print(f"✅ Patch prob mean: {self.patch_predictions.mean():.3f}")
        
    def _load_image(self, image_path: str) -> np.ndarray:
        """Orijinal görüntüyü yükler ve RGB'ye çevirir."""
        # Unicode path için numpy ile yükle
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Görüntü bulunamadı: {image_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        img_resized = cv2.resize(img_rgb, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
        return img_resized
    
    def _reshape_patch_predictions(self, patch_preds: np.ndarray) -> np.ndarray:
        """392 patch prediction'ı 14x28 grid'e dönüştürür."""
        return patch_preds.reshape(NUM_PATCHES_H, NUM_PATCHES_W)
    
    def visualize_single_image(
        self, 
        idx: int, 
        save: bool = True,
        show: bool = False
    ) -> plt.Figure:
        """
        Tek bir görüntü için patch attention map'i görselleştirir.
        
        Args:
            idx: Test set içindeki görüntü indeksi (0-73)
            save: Görüntüyü kaydet
            show: Görüntüyü göster
            
        Returns:
            matplotlib Figure
        """
        # Görüntü bilgilerini al
        pred_info = self.predictions[idx]
        image_path = pred_info['image_path']
        prediction = pred_info['prediction']
        target = pred_info['target']
        probability = pred_info['probability']
        
        # Patch predictions
        patch_preds = self.patch_predictions[idx]  # (392,)
        patch_grid = self._reshape_patch_predictions(patch_preds)  # (14, 28)
        
        # Görüntüyü yükle
        try:
            image_rgb = self._load_image(image_path)
        except FileNotFoundError:
            print(f"⚠️ Görüntü bulunamadı: {image_path}")
            return None
        
        # Normalize patch predictions using Z-score normalization
        # This amplifies even small variance within an image
        patch_mean = patch_grid.mean()
        patch_std = patch_grid.std()
        
        if patch_std < 0.001:  # Essentially no variance
            print(f"  ⚠️ Nearly zero variance (std={patch_std:.6f}), all patches same value")
            # All patches are same - just use a middle value
            patch_grid_norm = np.ones_like(patch_grid) * 0.5
        else:
            # Z-score normalization: (x - mean) / std
            patch_grid_zscore = (patch_grid - patch_mean) / patch_std
            
            # Map to [0, 1] range using sigmoid-like function
            # Values > mean → > 0.5, values < mean → < 0.5
            # Clip to ±3 std for visualization (covers 99.7% of data)
            patch_grid_clipped = np.clip(patch_grid_zscore, -3, 3)
            patch_grid_norm = (patch_grid_clipped + 3) / 6  # Map [-3, 3] → [0, 1]
            
            print(f"  📊 Patch stats: mean={patch_mean:.4f}, std={patch_std:.6f}, "
                  f"range=[{patch_grid.min():.4f}, {patch_grid.max():.4f}]")
        
        # OpenCV ile overlay oluştur
        overlay_img = image_rgb.copy()
        
        # Her patch için renkli dikdörtgen çiz
        for i in range(NUM_PATCHES_H):  # 14 rows
            for j in range(NUM_PATCHES_W):  # 28 cols
                patch_prob_norm = patch_grid_norm[i, j]
                
                # Patch koordinatları
                y1 = i * PATCH_SIZE
                y2 = (i + 1) * PATCH_SIZE
                x1 = j * PATCH_SIZE
                x2 = (j + 1) * PATCH_SIZE
                
                # Colormap'ten renk al (0-1 normalized değer)
                # RdYlGn_r: Red=1 (yüksek risk), Green=0 (düşük risk)
                color_normalized = self.cmap(patch_prob_norm)[:3]  # RGB values (0-1)
                color_bgr = tuple([int(c * 255) for c in reversed(color_normalized)])  # BGR for OpenCV
                
                # Semi-transparent overlay
                cv2.rectangle(overlay_img, (x1, y1), (x2, y2), color_bgr, -1)
        
        # Blend with original image
        alpha = 0.4
        blended = cv2.addWeighted(image_rgb, 1-alpha, overlay_img, alpha, 0)
        
        # Figure oluştur - 1 row, 2 columns
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # 1. Orijinal görüntü
        axes[0].imshow(image_rgb)
        axes[0].set_title(
            f"Original Panoramic X-Ray\n"
            f"Ground Truth: {'Fractured' if target == 1 else 'Healthy'} | "
            f"Prediction: {'Fractured' if prediction == 1 else 'Healthy'} ({probability:.1%})",
            fontsize=14, fontweight='bold'
        )
        axes[0].axis('off')
        
        # 2. Blended overlay
        axes[1].imshow(blended)
        axes[1].set_title(
            f"Patch-Level Attention Map (Z-score normalized)\n"
            f"14×28 grid = 392 patches (100×100 px each)\n"
            f"Mean: {patch_mean:.4f} | Std: {patch_std:.6f}\n"
            f"Red = Above Average Risk | Green = Below Average Risk",
            fontsize=12, fontweight='bold'
        )
        axes[1].axis('off')
        
        # Add colorbar - show normalized scale
        sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_label(f'Relative Risk (Z-score normalized)', rotation=270, labelpad=25, fontsize=10)
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(['-3σ', '-1.5σ', 'Mean', '+1.5σ', '+3σ'])
        
        # Genel başlık - prediction durumu
        if prediction == target:
            status_color = 'green'
            status_text = '✓ CORRECT PREDICTION'
        else:
            status_color = 'red'
            if prediction == 1 and target == 0:
                status_text = '✗ FALSE POSITIVE (Healthy predicted as Fractured)'
            else:
                status_text = '✗ FALSE NEGATIVE (Fractured predicted as Healthy)'
        
        image_name = Path(image_path).name
        fig.suptitle(
            f"Test Image #{idx+1}/74: {image_name}\n{status_text}",
            fontsize=16, fontweight='bold', color=status_color
        )
        
        plt.tight_layout()
        
        # Kaydet
        if save:
            save_name = f"test_{idx:03d}_{status_text.split()[1] if len(status_text.split()) > 1 else 'RESULT'}.png"
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Kaydedildi: {save_path}")
        
        # Göster
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def visualize_error_cases(
        self, 
        error_type: str = 'all',
        max_samples: int = 5
    ):
        """
        Hatalı prediction'ları görselleştirir.
        
        Args:
            error_type: 'false_positive', 'false_negative', 'all'
            max_samples: Maximum kaç örnek gösterilecek
        """
        print(f"\n🔍 Error Analysis: {error_type.upper()}")
        
        # Hataları bul
        errors = []
        for idx, pred_info in enumerate(self.predictions):
            pred = pred_info['prediction']
            target = pred_info['target']
            
            if pred != target:
                error_kind = 'false_positive' if (pred == 1 and target == 0) else 'false_negative'
                
                if error_type == 'all' or error_type == error_kind:
                    errors.append({
                        'idx': idx,
                        'type': error_kind,
                        'probability': pred_info['probability']
                    })
        
        print(f"📊 Toplam {len(errors)} hata bulundu")
        
        # En yüksek probability'ye göre sırala (en confident hatalar)
        errors_sorted = sorted(errors, key=lambda x: x['probability'], reverse=True)
        
        # İlk N tanesini görselleştir
        for i, error in enumerate(errors_sorted[:max_samples]):
            idx = error['idx']
            print(f"\n[{i+1}/{min(max_samples, len(errors))}] {error['type'].upper()}: Test image {idx+1}")
            print(f"  Probability: {error['probability']:.1%}")
            
            self.visualize_single_image(idx, save=True, show=False)
        
        print(f"\n✅ {min(max_samples, len(errors))} hata visualization'ı tamamlandı")
    
    def visualize_correct_predictions(
        self,
        class_label: Optional[int] = None,
        max_samples: int = 5
    ):
        """
        Doğru prediction'ları görselleştirir.
        
        Args:
            class_label: 0=Healthy, 1=Fractured, None=her ikisi
            max_samples: Maximum kaç örnek
        """
        print(f"\n✅ Correct Predictions Analysis")
        
        # Doğru prediction'ları bul
        correct = []
        for idx, pred_info in enumerate(self.predictions):
            pred = pred_info['prediction']
            target = pred_info['target']
            
            if pred == target:
                if class_label is None or target == class_label:
                    correct.append({
                        'idx': idx,
                        'class': target,
                        'probability': pred_info['probability']
                    })
        
        class_name = {0: 'Healthy', 1: 'Fractured', None: 'All'}.get(class_label, 'Unknown')
        print(f"📊 Toplam {len(correct)} doğru {class_name} prediction bulundu")
        
        # En yüksek confidence'a göre sırala
        correct_sorted = sorted(correct, key=lambda x: x['probability'], reverse=True)
        
        # İlk N tanesini görselleştir
        for i, item in enumerate(correct_sorted[:max_samples]):
            idx = item['idx']
            class_str = 'Fractured' if item['class'] == 1 else 'Healthy'
            print(f"\n[{i+1}/{min(max_samples, len(correct))}] CORRECT {class_str}: Test image {idx+1}")
            print(f"  Probability: {item['probability']:.1%}")
            
            self.visualize_single_image(idx, save=True, show=False)
        
        print(f"\n✅ {min(max_samples, len(correct))} correct prediction visualization'ı tamamlandı")
    
    def create_summary_grid(
        self,
        indices: List[int],
        grid_shape: Tuple[int, int] = (2, 4),
        save_name: str = "summary_grid.png"
    ):
        """
        Birden fazla görüntü için özet grid oluşturur.
        
        Args:
            indices: Görüntü indeksleri listesi
            grid_shape: Grid şekli (rows, cols)
            save_name: Kaydedilecek dosya adı
        """
        rows, cols = grid_shape
        fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
        axes = axes.flatten()
        
        for i, idx in enumerate(indices[:rows*cols]):
            pred_info = self.predictions[idx]
            image_path = pred_info['image_path']
            prediction = pred_info['prediction']
            target = pred_info['target']
            probability = pred_info['probability']
            
            # Patch predictions
            patch_preds = self.patch_predictions[idx]
            patch_grid = self._reshape_patch_predictions(patch_preds)
            
            # Görüntüyü yükle
            try:
                image = self._load_image(image_path)
            except FileNotFoundError:
                continue
            
            # Overlay
            axes[i].imshow(image, alpha=0.6)
            
            patch_grid_resized = cv2.resize(
                patch_grid, 
                (IMAGE_SIZE[1], IMAGE_SIZE[0]), 
                interpolation=cv2.INTER_LINEAR
            )
            
            axes[i].imshow(patch_grid_resized, cmap=self.cmap, alpha=0.5, vmin=0, vmax=1)
            
            # Başlık
            status = '✓' if prediction == target else '✗'
            pred_str = 'F' if prediction == 1 else 'H'
            target_str = 'F' if target == 1 else 'H'
            
            axes[i].set_title(
                f"{status} GT:{target_str} Pred:{pred_str} ({probability:.0%})",
                fontsize=10,
                color='green' if prediction == target else 'red'
            )
            axes[i].axis('off')
        
        # Boş subplot'ları gizle
        for i in range(len(indices), rows*cols):
            axes[i].axis('off')
        
        plt.suptitle("Test Set - Patch Attention Overview", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Kaydet
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Summary grid kaydedildi: {save_path}")
        plt.close(fig)


def main():
    """Ana fonksiyon."""
    parser = argparse.ArgumentParser(description="Patch Visualization Tool")
    parser.add_argument('--num_samples', type=int, default=10, help='Kaç örnek görselleştirilecek')
    parser.add_argument('--image_id', type=str, default=None, help='Spesifik görüntü adı')
    parser.add_argument('--error_analysis', action='store_true', help='Sadece hataları göster')
    parser.add_argument('--error_type', type=str, default='all', 
                       choices=['all', 'false_positive', 'false_negative'],
                       help='Hata tipi')
    parser.add_argument('--correct_only', action='store_true', help='Sadece doğru prediction\'ları göster')
    parser.add_argument('--class_label', type=int, default=None, choices=[0, 1],
                       help='Sınıf label: 0=Healthy, 1=Fractured')
    parser.add_argument('--summary_grid', action='store_true', help='Özet grid oluştur')
    
    args = parser.parse_args()
    
    # Visualizer oluştur
    visualizer = PatchVisualizer()
    
    print("\n" + "="*60)
    print("🎨 PATCH VISUALIZATION TOOL")
    print("="*60)
    
    if args.error_analysis:
        # Hata analizi
        visualizer.visualize_error_cases(
            error_type=args.error_type,
            max_samples=args.num_samples
        )
    
    elif args.correct_only:
        # Doğru prediction'lar
        visualizer.visualize_correct_predictions(
            class_label=args.class_label,
            max_samples=args.num_samples
        )
    
    elif args.summary_grid:
        # Özet grid - ilk N görüntü
        indices = list(range(min(args.num_samples, 8)))
        visualizer.create_summary_grid(indices, grid_shape=(2, 4))
    
    elif args.image_id:
        # Spesifik görüntü
        # TODO: Image ID ile arama
        print("⚠️ --image_id henüz implement edilmedi, indeks kullanın")
    
    else:
        # Default: İlk N görüntü
        print(f"\n📸 İlk {args.num_samples} test görüntüsü görselleştiriliyor...")
        for idx in range(min(args.num_samples, len(visualizer.predictions))):
            print(f"\n[{idx+1}/{args.num_samples}] Test image {idx+1}")
            visualizer.visualize_single_image(idx, save=True, show=False)
    
    print("\n" + "="*60)
    print(f"✅ Visualization tamamlandı!")
    print(f"📂 Çıktılar: {OUTPUT_DIR}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
