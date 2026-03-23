"""
Dogrulama: metadata'daki bbox bilgisi ile gercek crop dosyasini karsilastir
"""
import json
from pathlib import Path
from PIL import Image
import numpy as np

metadata_dir = Path('auto_labeled_crops/metadata')
crop_dir = Path('auto_labeled_crops/fractured')
sr_clahe_dir = Path('auto_labeled_crops_sr_clahe/fractured')

for stem in ['0283', '0326', '0191']:
    meta_path = metadata_dir / f'{stem}_tooth00.jpg.json'
    with open(meta_path) as f:
        meta = json.load(f)
    
    eb = meta['expanded_bbox']
    meta_w = eb['x_max'] - eb['x_min']
    meta_h = eb['y_max'] - eb['y_min']
    
    # Gercek crop dosyasi (auto_labeled_crops)
    crop_path = crop_dir / f'{stem}_tooth00.jpg'
    crop_exists = crop_path.exists()
    if crop_exists:
        crop_img = Image.open(crop_path)
        cw, ch = crop_img.size
    
    # SR+CLAHE versiyonu
    sr_path = sr_clahe_dir / f'{stem}_tooth00.jpg'
    sr_exists = sr_path.exists()
    if sr_exists:
        sr_img = Image.open(sr_path)
        sw, sh = sr_img.size
    
    print(f"=== {stem}_tooth00 ===")
    print(f"  Metadata expanded_bbox: {meta_w} x {meta_h}")
    print(f"  auto_labeled_crops/fractured: {'VAR' if crop_exists else 'YOK'}", end="")
    if crop_exists:
        print(f" -> {cw} x {ch}", end="")
        print(f" {'ESLESIYOR' if (cw==meta_w and ch==meta_h) else 'FARKLI!'}")
    else:
        print()
    print(f"  auto_labeled_crops_sr_clahe/fractured: {'VAR' if sr_exists else 'YOK'}", end="")
    if sr_exists:
        print(f" -> {sw} x {sh}")
        print(f"  SR 4x faktor: {sw/cw:.1f}x genislik, {sh/ch:.1f}x yukseklik" if crop_exists else "")
    else:
        print()
    print()
