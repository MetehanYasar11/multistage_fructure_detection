"""
Pipeline Visualization v4:
- Son panel: Fracture line'ın minimum enclosing bbox crop'u (margin=5px)
- Başlık kayması düzeltildi
- 3 farklı fractured örnek
"""
import cv2
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json

fractured_dir = Path('okandataset_final/Dataset/Fractured')
output_dir = Path('outputs/pipeline_figure')
output_dir.mkdir(parents=True, exist_ok=True)

EXAMPLES = [
    ('0283', 0),
    ('0326', 0),
    ('0191', 0),
]

def make_pipeline_figure(stem, tooth_idx, out_name):
    meta_path = f'auto_labeled_crops/metadata/{stem}_tooth{tooth_idx:02d}.jpg.json'
    with open(meta_path) as f:
        meta = json.load(f)

    eb = meta['expanded_bbox']
    fl = meta['fracture_lines_in_bbox'][0]

    img_path = fractured_dir / f'{stem}.jpg'
    img_bgr = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]

    crop_w = eb['x_max'] - eb['x_min']
    crop_h = eb['y_max'] - eb['y_min']

    # Minimum enclosing bbox of fracture line (sadece çizgiyi saran en küçük kutu)
    # margin=5px — gerçekten ne kadar küçük olduğunu göstermek için
    margin = 5
    meb_x1 = max(0, int(min(fl['x1'], fl['x2'])) - margin)
    meb_y1 = max(0, int(min(fl['y1'], fl['y2'])) - margin)
    meb_x2 = min(W, int(max(fl['x1'], fl['x2'])) + margin)
    meb_y2 = min(H, int(max(fl['y1'], fl['y2'])) + margin)
    meb_w = meb_x2 - meb_x1
    meb_h = meb_y2 - meb_y1
    
    fl_len = ((fl['x2']-fl['x1'])**2 + (fl['y2']-fl['y1'])**2)**0.5

    crop_img = img_rgb[eb['y_min']:eb['y_max'], eb['x_min']:eb['x_max']].copy()
    frac_img = img_rgb[meb_y1:meb_y2, meb_x1:meb_x2].copy()

    print(f'  Full={W}x{H}  Crop={crop_w}x{crop_h}  '
          f'FracBBox={meb_w}x{meb_h}  Line={fl_len:.1f}px')

    # ── FIGURE: 5 columns ──
    fig, axes = plt.subplots(1, 5, figsize=(22, 6),
                             gridspec_kw={'width_ratios': [5, 0.55, 1.8, 0.55, 1.5],
                                          'wspace': 0.08},
                             facecolor='white')

    # ═══ (a) Full panoramic ═══
    ax1 = axes[0]
    ax1.imshow(img_rgb)
    # RCT bbox (cyan)
    ax1.add_patch(patches.Rectangle(
        (eb['x_min'], eb['y_min']), crop_w, crop_h,
        lw=2.5, edgecolor='#00e5ff', facecolor='none', zorder=5))
    # Fracture line on full
    ax1.plot([fl['x1'], fl['x2']], [fl['y1'], fl['y2']],
             color='#ff1744', lw=2.5, marker='o', ms=4, zorder=6)
    ax1.set_title(f'(a)  Input Panoramic Radiograph\n{W} × {H} px',
                  fontsize=12, fontweight='bold', loc='center', pad=8)
    ax1.set_axis_off()

    # ═══ Arrow 1 ═══
    ax_a1 = axes[1]
    ax_a1.set_xlim(0, 1); ax_a1.set_ylim(0, 1)
    ax_a1.annotate('', xy=(0.85, 0.5), xytext=(0.15, 0.5),
                   arrowprops=dict(arrowstyle='-|>', color='#37474f', lw=2.5,
                                   mutation_scale=20))
    ax_a1.text(0.5, 0.66, 'Stage 1', ha='center', fontsize=10,
               fontweight='bold', color='#37474f')
    ax_a1.text(0.5, 0.58, 'YOLO RCT', ha='center', fontsize=8, color='#78909c')
    ax_a1.text(0.5, 0.36, 'crop &\nrescale', ha='center', fontsize=8,
               style='italic', color='#90a4ae')
    ax_a1.set_axis_off()

    # ═══ (b) RCT crop ═══
    ax2 = axes[2]
    ax2.imshow(crop_img)
    # Fracture line on crop
    fl_x1r = fl['x1'] - eb['x_min']
    fl_y1r = fl['y1'] - eb['y_min']
    fl_x2r = fl['x2'] - eb['x_min']
    fl_y2r = fl['y2'] - eb['y_min']
    # Min enclosing bbox on crop (red dashed)
    meb_x1r = meb_x1 - eb['x_min']
    meb_y1r = meb_y1 - eb['y_min']
    ax2.add_patch(patches.Rectangle(
        (meb_x1r, meb_y1r), meb_w, meb_h,
        lw=2, edgecolor='#ff1744', facecolor='#ff174415', linestyle='--', zorder=5))
    ax2.plot([fl_x1r, fl_x2r], [fl_y1r, fl_y2r],
             color='#ff1744', lw=3, marker='o', ms=6, zorder=6)
    ax2.set_title(f'(b)  Detected RCT Tooth\n{crop_w} × {crop_h} px',
                  fontsize=12, fontweight='bold', loc='center', pad=8)
    ax2.set_axis_off()

    # ═══ Arrow 2 ═══
    ax_a2 = axes[3]
    ax_a2.set_xlim(0, 1); ax_a2.set_ylim(0, 1)
    ax_a2.annotate('', xy=(0.85, 0.5), xytext=(0.15, 0.5),
                   arrowprops=dict(arrowstyle='-|>', color='#37474f', lw=2.5,
                                   mutation_scale=20))
    ax_a2.text(0.5, 0.66, 'Stage 2', ha='center', fontsize=10,
               fontweight='bold', color='#37474f')
    ax_a2.text(0.5, 0.58, 'ViT Classifier', ha='center', fontsize=8, color='#78909c')
    ax_a2.text(0.5, 0.36, 'fracture\ndetection', ha='center', fontsize=8,
               style='italic', color='#90a4ae')
    ax_a2.set_axis_off()

    # ═══ (c) Fracture min enclosing bbox ═══
    ax3 = axes[4]
    ax3.imshow(frac_img)
    # Fracture line on zoom
    fl_x1z = fl['x1'] - meb_x1
    fl_y1z = fl['y1'] - meb_y1
    fl_x2z = fl['x2'] - meb_x1
    fl_y2z = fl['y2'] - meb_y1
    ax3.plot([fl_x1z, fl_x2z], [fl_y1z, fl_y2z],
             color='#ff1744', lw=4, marker='o', ms=8, zorder=6)
    ax3.set_title(f'(c)  Fracture GT BBox\n{meb_w} × {meb_h} px',
                  fontsize=12, fontweight='bold', loc='center', pad=8, color='#d50000')
    ax3.set_axis_off()

    # ═══ Bottom stats ═══
    crop_area_pct = (crop_w * crop_h) / (W * H) * 100
    frac_area_pct = (meb_w * meb_h) / (W * H) * 100
    fig.text(0.5, 0.01,
             f'Area:  Full = {W*H:,} px²   │   '
             f'RCT crop = {crop_w*crop_h:,} px² ({crop_area_pct:.1f}%)   │   '
             f'Fracture GT bbox = {meb_w*meb_h:,} px² ({frac_area_pct:.3f}%)   │   '
             f'Fracture line ≈ {fl_len:.0f} px',
             ha='center', fontsize=10, color='#555',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#fafafa', edgecolor='#ddd'))

    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.08)

    for suffix, dpi in [('.png', 200), ('_hr.png', 350)]:
        out = output_dir / f'{out_name}{suffix}'
        fig.savefig(str(out), dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none', pad_inches=0.15)
        print(f'  Saved: {out}')

    plt.close(fig)


for stem, tidx in EXAMPLES:
    print(f'\n=== {stem} (tooth{tidx:02d}) ===')
    make_pipeline_figure(stem, tidx, f'pipeline_{stem}')

print('\nAll done!')
