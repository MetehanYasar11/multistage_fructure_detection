"""
Stage 2 Fracture Detection Visualization - Streamlit App

Stage 2 (fracture detection) sonuçlarını görselleştirir:
- Test dataseti crop'larını gösterir
- stage2_detector_nano.pt modeli ile fracture detection yapar
- Ground truth vs prediction karşılaştırması

Author: Master's Thesis Project
Date: November 26, 2025
"""

import streamlit as st
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO

# Monkey patch ultralytics
import ultralytics.nn.tasks as tasks_module

def patched_torch_safe_load(file):
    """Patched version that uses weights_only=False"""
    try:
        ckpt = torch.load(file, map_location="cpu", weights_only=False)
        return ckpt, file
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise

tasks_module.torch_safe_load = patched_torch_safe_load


# Page config
st.set_page_config(
    page_title="Stage 2: Fracture Detection",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    /* Navigation buttons */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
        padding: 10px;
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] .stRadio > label {
        font-size: 18px !important;
        font-weight: bold !important;
        padding: 10px !important;
    }
    </style>
""", unsafe_allow_html=True)

# Add keyboard navigation
st.components.v1.html(
    """
    <script>
    const doc = window.parent.document;
    doc.removeEventListener('keydown', handleKeyPress);
    
    function handleKeyPress(e) {
        if (e.keyCode === 37) {
            const buttons = doc.querySelectorAll('button');
            for (let btn of buttons) {
                if (btn.innerText.includes('Previous') || btn.innerText.includes('⬅️')) {
                    btn.click();
                    break;
                }
            }
        } else if (e.keyCode === 39) {
            const buttons = doc.querySelectorAll('button');
            for (let btn of buttons) {
                if (btn.innerText.includes('Next') || btn.innerText.includes('➡️')) {
                    btn.click();
                    break;
                }
            }
        }
    }
    
    doc.addEventListener('keydown', handleKeyPress);
    </script>
    """,
    height=0,
)

st.title("🔍 Stage 2: Fracture Detection Visualization")
st.markdown("**Fracture Predictions** (yeşil) vs **Ground Truth** (kırmızı)")

st.info(
    "### 🎯 Quick Start\n"
    "1️⃣ Model ve test dataseti otomatik yüklenir\n"
    "2️⃣ ⬅️ ➡️ ok tuşları ile navigate edin\n"
    "3️⃣ Sidebar'dan threshold ayarlayın\n"
    "4️⃣ Ground truth ve prediction'ları karşılaştırın"
)


@st.cache_resource
def load_yolo_model():
    """Load Stage 2 YOLO model"""
    detector_path = Path("detectors/stage2_detector_nano.pt")
    if not detector_path.exists():
        st.error(f"Detector not found: {detector_path}")
        return None
    return YOLO(str(detector_path))


@st.cache_data
def get_test_images():
    """Get all test images and labels"""
    test_images_dir = Path("stage2_fracture_dataset/test/images")
    test_labels_dir = Path("stage2_fracture_dataset/test/labels")
    
    if not test_images_dir.exists():
        st.error(f"Test images directory not found: {test_images_dir}")
        return []
    
    images = sorted(test_images_dir.glob("*.png"))
    
    # Pair each image with its label file
    image_label_pairs = []
    for img_path in images:
        label_path = test_labels_dir / f"{img_path.stem}.txt"
        image_label_pairs.append({
            'image': img_path,
            'label': label_path if label_path.exists() else None,
            'name': img_path.name
        })
    
    return image_label_pairs


def load_ground_truth_boxes(label_path, img_width, img_height):
    """Load ground truth boxes from YOLO format label file"""
    if not label_path or not label_path.exists():
        return []
    
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls, cx, cy, w, h = [float(x) for x in parts[:5]]
                
                # Convert from normalized YOLO format to pixel coordinates
                cx_px = cx * img_width
                cy_px = cy * img_height
                w_px = w * img_width
                h_px = h * img_height
                
                x1 = cx_px - w_px / 2
                y1 = cy_px - h_px / 2
                x2 = cx_px + w_px / 2
                y2 = cy_px + h_px / 2
                
                boxes.append({
                    'class': int(cls),
                    'bbox': [x1, y1, x2, y2],
                    'center': [cx_px, cy_px],
                    'size': [w_px, h_px]
                })
    
    return boxes


def detect_fractures(model, image_path, conf_threshold=0.25, iou_threshold=0.45):
    """Detect fractures using Stage 2 model"""
    results = model.predict(
        source=str(image_path),
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False
    )
    
    detections = []
    
    if len(results) > 0:
        result = results[0]
        boxes = result.boxes
        
        for i in range(len(boxes)):
            conf = float(boxes.conf[i])
            bbox = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
            cls = int(boxes.cls[i])
            
            detections.append({
                'bbox': bbox.tolist(),
                'conf': conf,
                'class': cls
            })
    
    return detections


def draw_visualization(image, predictions, ground_truth, show_pred, show_gt, bbox_alpha):
    """Draw predictions and ground truth on image"""
    # Convert to RGBA for transparency
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        small_font = ImageFont.truetype("arial.ttf", 10)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Draw ground truth (RED)
    if show_gt and ground_truth:
        for gt_box in ground_truth:
            x1, y1, x2, y2 = gt_box['bbox']
            
            # Red semi-transparent fill
            alpha_val = int(255 * bbox_alpha)
            draw.rectangle(
                [x1, y1, x2, y2],
                outline=(255, 0, 0, 255),  # Red border
                fill=(255, 0, 0, alpha_val),  # Red fill
                width=2
            )
            
            # Label
            draw.text((x1, y1 - 15), "GT", fill=(255, 0, 0, 255), font=small_font)
    
    # Draw predictions (GREEN)
    if show_pred and predictions:
        for pred in predictions:
            x1, y1, x2, y2 = pred['bbox']
            conf = pred['conf']
            
            # Green semi-transparent fill
            alpha_val = int(255 * bbox_alpha)
            draw.rectangle(
                [x1, y1, x2, y2],
                outline=(0, 255, 0, 255),  # Green border
                fill=(0, 255, 0, alpha_val),  # Green fill
                width=2
            )
            
            # Confidence label
            label = f"Fracture {conf:.2f}"
            draw.text((x1, y1 - 15), label, fill=(0, 255, 0, 255), font=small_font)
    
    # Composite
    result = Image.alpha_composite(image, overlay)
    return result.convert('RGB')


def main():
    # Load model
    with st.spinner("Loading Stage 2 model..."):
        model = load_yolo_model()
    
    if model is None:
        st.stop()
    
    st.sidebar.success("✅ Model loaded: stage2_detector_nano.pt")
    
    # Load test images
    test_data = get_test_images()
    
    if len(test_data) == 0:
        st.error("No test images found!")
        st.stop()
    
    st.sidebar.markdown("---")
    st.sidebar.subheader(f"📁 Test Dataset")
    st.sidebar.write(f"Total images: {len(test_data)}")
    
    # Initialize session state
    if 'current_image_idx' not in st.session_state:
        st.session_state.current_image_idx = 0
    
    # Ensure index is valid
    if st.session_state.current_image_idx >= len(test_data):
        st.session_state.current_image_idx = 0
    
    current_idx = st.session_state.current_image_idx
    
    st.sidebar.markdown("---")
    
    # Image selector
    image_names = [item['name'] for item in test_data]
    selected_image_name = st.sidebar.selectbox(
        "Select Image",
        image_names,
        index=current_idx
    )
    
    # Update current index
    st.session_state.current_image_idx = image_names.index(selected_image_name)
    current_idx = st.session_state.current_image_idx
    
    st.sidebar.markdown("---")
    
    # Display options
    st.sidebar.subheader("Display Options")
    show_pred = st.sidebar.checkbox("Show Predictions", value=True)
    show_gt = st.sidebar.checkbox("Show Ground Truth", value=True)
    bbox_alpha = st.sidebar.slider("Bbox Transparency", 0.0, 1.0, 0.3, 0.05)
    
    st.sidebar.markdown("---")
    
    # Detection thresholds
    st.sidebar.subheader("Detection Thresholds")
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    iou_threshold = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05)
    
    # Load current image
    current_data = test_data[current_idx]
    image_path = current_data['image']
    label_path = current_data['label']
    
    image = Image.open(image_path)
    img_width, img_height = image.size
    
    # Load ground truth
    ground_truth = load_ground_truth_boxes(label_path, img_width, img_height)
    
    # Detect fractures
    with st.spinner("Detecting fractures..."):
        predictions = detect_fractures(model, image_path, conf_threshold, iou_threshold)
    
    # Show statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Predictions", len(predictions))
    with col2:
        st.metric("Ground Truth", len(ground_truth))
    with col3:
        avg_conf = np.mean([p['conf'] for p in predictions]) if predictions else 0.0
        st.metric("Avg Confidence", f"{avg_conf:.3f}")
    with col4:
        has_gt = "✅ Yes" if ground_truth else "❌ No"
        st.metric("Has GT", has_gt)
    
    # Navigation
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        if current_idx > 0:
            if st.button("⬅️ Previous"):
                st.session_state.current_image_idx -= 1
                st.rerun()
    
    with col2:
        st.write(f"**Image {current_idx + 1} / {len(test_data)}**")
    
    with col3:
        if current_idx < len(test_data) - 1:
            if st.button("➡️ Next"):
                st.session_state.current_image_idx += 1
                st.rerun()
    
    # Draw visualization
    result_image = draw_visualization(
        image, predictions, ground_truth,
        show_pred, show_gt, bbox_alpha
    )
    
    # Display - fixed width for crops
    st.image(result_image, caption=f"{current_data['name']}", width=800)
    
    # Detailed info
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Current Image Stats")
    st.sidebar.write(f"**Name:** {current_data['name']}")
    st.sidebar.write(f"**Size:** {img_width}x{img_height}")
    st.sidebar.write(f"**Predictions:** {len(predictions)}")
    st.sidebar.write(f"**Ground Truth:** {len(ground_truth)}")
    
    if predictions:
        st.sidebar.markdown("**Prediction Details:**")
        for i, pred in enumerate(predictions):
            st.sidebar.write(f"  {i+1}. Conf: {pred['conf']:.3f}")
    
    # Legend
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎨 Legend")
    st.sidebar.markdown("🟢 **Green Box**: Prediction")
    st.sidebar.markdown("   - Model detected fracture")
    st.sidebar.markdown("   - Confidence score on top")
    st.sidebar.markdown("🔴 **Red Box**: Ground Truth")
    st.sidebar.markdown("   - Actual fracture location")
    st.sidebar.markdown("   - From manual annotation")


if __name__ == "__main__":
    main()
