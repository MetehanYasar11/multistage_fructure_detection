"""
KRITIK GORSEL KANITLAR:
Orphan fracture cizgilerinin Healthy resimlere ait olup olmadigini
gorsellestirerek kontrol et.

Eger cizgiler gercekten resme ait degilse:
- Dis bolgesine denk gelmez
- Rastgele konuma duser
- Ya da farkli buyuklukte bir resme aittir
"""

import cv2
import numpy as np
from pathlib import Path

fractured_dir = Path('okandataset_final/Dataset/Fractured')
healthy_dir = Path('okandataset_final/Dataset/Healthy')
output_dir = Path('outputs/orphan_verification')
output_dir.mkdir(parents=True, exist_ok=True)

def load_lines_from_txt(txt_path):
    lines = []
    with open(txt_path) as f:
        content = f.read().strip().split('\n')
    for i in range(0, len(content), 2):
        if i + 1 < len(content):
            try:
                p1 = [float(x) for x in content[i].split()]
                p2 = [float(x) for x in content[i+1].split()]
                if len(p1) >= 2 and len(p2) >= 2:
                    lines.append((p1[0], p1[1], p2[0], p2[1]))
            except:
                pass
    return lines

# Birkaç orphan resmi görselleştir
test_stems = ['0002', '0004', '0007', '0010', '0016', '0023']

for stem in test_stems:
    img_path = healthy_dir / f'{stem}.jpg'
    if not img_path.exists():
        continue
    
    # Resmi oku (Türkçe karakter sorunu için)
    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        continue
    
    h, w = img.shape[:2]
    
    # Healthy GT çizgileri (yeşil - DOĞRU etiketler)
    h_txt = healthy_dir / f'{stem}.txt'
    if h_txt.exists():
        h_lines = load_lines_from_txt(h_txt)
        for line in h_lines:
            x1, y1, x2, y2 = [int(v) for v in line]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.circle(img, (x1, y1), 6, (0, 255, 0), -1)
            cv2.circle(img, (x2, y2), 6, (0, 255, 0), -1)
    
    # Orphan Fractured GT çizgileri (kırmızı - SORU İŞARETİ)
    f_txt = fractured_dir / f'{stem}.txt'
    if f_txt.exists():
        f_lines = load_lines_from_txt(f_txt)
        for line in f_lines:
            x1, y1, x2, y2 = [int(v) for v in line]
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 4)
            cv2.circle(img, (x1, y1), 8, (0, 0, 255), -1)
            cv2.circle(img, (x2, y2), 8, (0, 0, 255), -1)
            # Kırmızı çizginin yanına "ORPHAN FRACTURE?" yaz
            cv2.putText(img, "ORPHAN FRAC?", (x1-100, y1-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Başlık ekle
    cv2.putText(img, f"Healthy/{stem}.jpg - GREEN=Healthy GT, RED=Orphan Fractured GT",
               (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    cv2.putText(img, f"Soru: Kirmizi cizgi bu resme mi ait?",
               (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    # Zoom: Kırmızı çizgi bölgesine yakınlaştır
    if f_txt.exists() and f_lines:
        fl = f_lines[0]
        cx = int((fl[0] + fl[2]) / 2)
        cy = int((fl[1] + fl[3]) / 2)
        margin = 200
        
        x1_crop = max(0, cx - margin)
        y1_crop = max(0, cy - margin)
        x2_crop = min(w, cx + margin)
        y2_crop = min(h, cy + margin)
        
        zoom = img[y1_crop:y2_crop, x1_crop:x2_crop].copy()
        zoom = cv2.resize(zoom, (400, 400))
        
        # Zoom'u sağ üst köşeye ekle
        img[10:410, w-410:w-10] = zoom
        cv2.rectangle(img, (w-410, 10), (w-10, 410), (0, 0, 255), 3)
        cv2.putText(img, "ZOOM: Orphan Frac", (w-400, 430),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Kaydet
    out_path = output_dir / f'{stem}_orphan_verify.jpg'
    is_success, im_buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if is_success:
        im_buf.tofile(str(out_path))
        print(f"Saved: {out_path}")

print("\nToplam:", len(test_stems), "resim kaydedildi")
