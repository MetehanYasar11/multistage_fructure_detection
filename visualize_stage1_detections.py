"""
Stage 1 Detection Visualization - Streamlit App

RCT detection bboxları (şeffaf yeşil) ve ground truth fracture lines (kırmızı) 
birlikte görselleştirir.

Author: Master's Thesis Project
Date: November 25, 2025
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
    page_title="Stage 1: RCT Detection",
    page_icon="🦷",
    layout="wide"
)

# Custom CSS for VERY VISIBLE drag & drop UI
st.markdown("""
    <style>
    /* Make file uploader VERY visible */
    [data-testid="stFileUploader"] {
        border: 4px dashed #2196F3 !important;
        border-radius: 15px !important;
        padding: 30px !important;
        background: linear-gradient(135deg, #f0f7ff 0%, #e3f2fd 100%) !important;
        transition: all 0.3s ease !important;
        min-height: 200px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #1976D2 !important;
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%) !important;
        transform: scale(1.02) !important;
        box-shadow: 0 8px 16px rgba(33, 150, 243, 0.3) !important;
    }
    
    /* Uploaded file success state */
    .uploadedFile {
        border: 3px solid #4CAF50 !important;
        border-radius: 10px !important;
        padding: 15px !important;
        background: linear-gradient(135deg, #f0f8f0 0%, #e8f5e9 100%) !important;
        margin: 10px 0 !important;
    }
    
    /* Make browse button bigger */
    [data-testid="stFileUploader"] button {
        font-size: 16px !important;
        padding: 12px 24px !important;
        font-weight: bold !important;
        background-color: #2196F3 !important;
        color: white !important;
        border-radius: 8px !important;
    }
    
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
    
    /* Sidebar radio buttons */
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
    
    // Remove old listeners to prevent duplicates
    doc.removeEventListener('keydown', handleKeyPress);
    
    function handleKeyPress(e) {
        // Left arrow (37) or Right arrow (39)
        if (e.keyCode === 37) {
            // Find and click the Previous button
            const buttons = doc.querySelectorAll('button');
            for (let btn of buttons) {
                if (btn.innerText.includes('Previous') || btn.innerText.includes('⬅️')) {
                    btn.click();
                    break;
                }
            }
        } else if (e.keyCode === 39) {
            // Find and click the Next button
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

st.title("🦷 Stage 1: RCT Detection Visualization")
st.markdown("**RCT Bboxes** (şeffaf yeşil) + **Ground Truth Fracture Lines** (kırmızı)")

# Big prominent info box
st.info(
    "### 🎯 Quick Start\n"
    "1️⃣ **Select Mode** from sidebar (📁 Upload or 💾 Default Dataset)\n"
    "2️⃣ **Drag & Drop** your images into upload areas\n"
    "3️⃣ **Navigate** with ⬅️ ➡️ arrow keys or buttons\n"
    "4️⃣ **Adjust** settings in sidebar"
)


@st.cache_resource
def load_yolo_model():
    """Load YOLO model"""
    # Try Docker path first, fallback to relative path
    docker_detector = Path("/workspace/detectors/RCTdetector_v11x.pt")
    local_detector = Path("detectors/RCTdetector_v11x.pt")
    
    if docker_detector.exists():
        detector_path = docker_detector
    elif local_detector.exists():
        detector_path = local_detector
    else:
        st.error(f"Detector not found in {docker_detector} or {local_detector}")
        return None
    
    return YOLO(str(detector_path))


@st.cache_data
def load_test_split():
    """Load test split indices"""
    # Try Docker path first, fallback to relative path
    docker_split = Path("/workspace/vision_transformer/outputs/splits/train_val_test_split.json")
    local_split = Path("vision_transformer/outputs/splits/train_val_test_split.json")
    
    if docker_split.exists():
        split_path = docker_split
    elif local_split.exists():
        split_path = local_split
    else:
        st.error(f"Split file not found in {docker_split} or {local_split}")
        return []
    
    with open(split_path, 'r') as f:
        splits = json.load(f)
    return splits.get('test', [])


@st.cache_data
def get_test_fractured_images():
    """Get list of fractured images from test split"""
    # Try Docker path first, fallback to Windows path
    docker_dataset = Path("/workspace/data/Fractured")
    windows_dataset = Path(r"C:\Users\maspe\OneDrive\Masaüstü\masterthesis\Dataset_2021\Dataset_2021\Dataset\Fractured")
    
    if docker_dataset.exists():
        fractured_dir = docker_dataset
    elif windows_dataset.exists():
        fractured_dir = windows_dataset
    else:
        st.error(f"Dataset not found in Docker ({docker_dataset}) or Windows ({windows_dataset})")
        return []
    
    # Get all fractured images
    all_fractured = sorted(fractured_dir.glob("*.png"))
    
    # Load test split
    test_indices = load_test_split()
    
    # Filter only test images
    test_fractured = []
    for img_path in all_fractured:
        # Extract index from filename (e.g., "0053.png" -> 53)
        idx = int(img_path.stem)
        if idx in test_indices:
            test_fractured.append((idx, img_path))
    
    return test_fractured


def load_ground_truth_lines(idx):
    """Load ground truth fracture lines from file path"""
    # Try Docker path first, fallback to Windows path
    docker_annotation = Path(f"/workspace/data/Fractured/{idx:04d}.txt")
    windows_annotation = Path(rf"C:\Users\maspe\OneDrive\Masaüstü\masterthesis\Dataset_2021\Dataset_2021\Dataset\Fractured\{idx:04d}.txt")
    
    if docker_annotation.exists():
        annotation_path = docker_annotation
    elif windows_annotation.exists():
        annotation_path = windows_annotation
    else:
        return []
    
    lines = []
    with open(annotation_path, 'r') as f:
        content = f.read().strip().split('\n')
        
        # Each line has 2 points
        for i in range(0, len(content), 2):
            if i + 1 < len(content):
                try:
                    point1 = [float(x) for x in content[i].split()]
                    point2 = [float(x) for x in content[i + 1].split()]
                    
                    x1, y1 = point1[0], point1[1]
                    x2, y2 = point2[0], point2[1]
                    
                    lines.append((x1, y1, x2, y2))
                except:
                    continue
    
    return lines


def load_ground_truth_from_uploaded_file(uploaded_file):
    """Load ground truth fracture lines from uploaded file"""
    if uploaded_file is None:
        return []
    
    lines = []
    try:
        # Read uploaded file content
        content = uploaded_file.read().decode('utf-8').strip().split('\n')
        
        # Each line has 2 points
        for i in range(0, len(content), 2):
            if i + 1 < len(content):
                try:
                    point1 = [float(x) for x in content[i].split()]
                    point2 = [float(x) for x in content[i + 1].split()]
                    
                    x1, y1 = point1[0], point1[1]
                    x2, y2 = point2[0], point2[1]
                    
                    lines.append((x1, y1, x2, y2))
                except:
                    continue
        
        # Reset file pointer for potential re-reading
        uploaded_file.seek(0)
    except Exception as e:
        st.warning(f"Error reading annotation: {e}")
    
    return lines


def detect_rct(model, image_path, conf_threshold=0.25, iou_threshold=0.45, scale_factor=2.0):
    """Detect RCT teeth and expand bboxes by scale_factor"""
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
        
        # RCT class index = 9
        rct_class_idx = 9
        
        for i in range(len(boxes)):
            cls = int(boxes.cls[i])
            
            if cls == rct_class_idx:
                conf = float(boxes.conf[i])
                bbox = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                
                # Expand bbox by scale_factor (2x = double the size)
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                
                # Calculate center
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                # Scale width and height
                new_width = width * scale_factor
                new_height = height * scale_factor
                
                # Calculate new bbox
                x1_new = cx - new_width / 2
                y1_new = cy - new_height / 2
                x2_new = cx + new_width / 2
                y2_new = cy + new_height / 2
                
                # Clip to image bounds (0 to image size)
                x1_new = max(0, x1_new)
                y1_new = max(0, y1_new)
                x2_new = max(0, x2_new)
                y2_new = max(0, y2_new)
                
                detections.append({
                    'bbox': [x1_new, y1_new, x2_new, y2_new],
                    'conf': conf
                })
    
    return detections


def draw_visualization(image, detections, ground_truth_lines, show_rct=True, show_gt=True, bbox_alpha=0.3):
    """Draw RCT bboxes and ground truth lines"""
    # Convert to RGBA for transparency
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # Create overlay for transparent boxes
    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    
    # Create draw object for lines
    draw = ImageDraw.Draw(image)
    
    # Draw RCT bboxes (transparent green)
    if show_rct and detections:
        for det in detections:
            bbox = det['bbox']
            conf = det['conf']
            
            x1, y1, x2, y2 = bbox
            
            # Transparent green fill
            green_alpha = int(255 * bbox_alpha)
            draw_overlay.rectangle(
                [x1, y1, x2, y2],
                fill=(0, 255, 0, green_alpha)
            )
            
            # Green border (not transparent)
            draw.rectangle(
                [x1, y1, x2, y2],
                outline=(0, 255, 0),
                width=3
            )
            
            # Confidence text
            text = f"{conf:.2f}"
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            # Text background
            text_bbox = draw.textbbox((x1, y1 - 25), text, font=font)
            draw.rectangle(text_bbox, fill=(0, 255, 0))
            draw.text((x1, y1 - 25), text, fill=(0, 0, 0), font=font)
    
    # Composite overlay with image
    image = Image.alpha_composite(image, overlay)
    
    # Convert back to RGB for drawing lines
    image = image.convert('RGB')
    draw = ImageDraw.Draw(image)
    
    # Draw ground truth lines (red)
    if show_gt and ground_truth_lines:
        for line in ground_truth_lines:
            x1, y1, x2, y2 = line
            
            # Thick red line
            draw.line(
                [(x1, y1), (x2, y2)],
                fill=(255, 0, 0),
                width=5
            )
            
            # Draw circles at endpoints
            radius = 8
            draw.ellipse(
                [x1 - radius, y1 - radius, x1 + radius, y1 + radius],
                fill=(255, 0, 0),
                outline=(255, 255, 255),
                width=2
            )
            draw.ellipse(
                [x2 - radius, y2 - radius, x2 + radius, y2 + radius],
                fill=(255, 0, 0),
                outline=(255, 255, 255),
                width=2
            )
    
    return image


def main():
    """Main app"""
    
    # Load model
    model = load_yolo_model()
    
    if model is None:
        st.stop()
    
    # Sidebar header with big mode selection
    st.sidebar.title("🎛️ Controls")
    
    # Mode selection - BIG and PROMINENT
    st.sidebar.markdown("## 🚀 Select Mode")
    mode = st.sidebar.radio(
        "",
        ["📁 Upload Images", "💾 Use Default Dataset"],
        index=0,
        help="Choose upload to use your own images, or default to browse existing dataset"
    )
    
    if mode == "📁 Upload Images":
        st.sidebar.success("✅ Upload Mode Active")
        st.sidebar.markdown("👇 Upload your images below in the main area")
    else:
        st.sidebar.info("✅ Dataset Mode Active")
    
    st.sidebar.markdown("---")
    
    # Initialize session state
    if 'uploaded_images' not in st.session_state:
        st.session_state.uploaded_images = []
    if 'uploaded_annotations' not in st.session_state:
        st.session_state.uploaded_annotations = {}
    if 'current_image_idx' not in st.session_state:
        st.session_state.current_image_idx = 0
    
    # Get images based on mode
    if mode == "📁 Upload Images":
        st.sidebar.subheader("📤 Upload Files")
        
        # Main upload area with drag & drop - VERY VISIBLE
        st.markdown("---")
        st.markdown("# 🎯 Drag & Drop Your Files Here!")
        st.markdown("### Or click 'Browse files' to select from computer")
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📸 Step 1: Upload Images")
            st.markdown("**Supported:** PNG, JPG, JPEG")
            uploaded_files = st.file_uploader(
                "Drag & drop images here or click to browse",
                type=['png', 'jpg', 'jpeg'],
                accept_multiple_files=True,
                key="image_uploader",
                label_visibility="collapsed"
            )
            
            if uploaded_files:
                st.session_state.uploaded_images = uploaded_files
                st.success(f"✅ {len(uploaded_files)} images uploaded")
        
        with col2:
            st.markdown("### 📝 Step 2: Upload Ground Truth (Optional)")
            st.markdown("**Format:** .txt files with same names as images")
            uploaded_annotations = st.file_uploader(
                "Drag & drop .txt files here or click to browse",
                type=['txt'],
                accept_multiple_files=True,
                key="annotation_uploader",
                help="Upload .txt files with same names as images (e.g., 0001.txt for 0001.png)",
                label_visibility="collapsed"
            )
            
            if uploaded_annotations:
                # Store annotations by filename (without extension)
                st.session_state.uploaded_annotations = {}
                matched_count = 0
                
                for ann_file in uploaded_annotations:
                    # Remove .txt extension to get base name
                    base_name = ann_file.name.replace('.txt', '')
                    st.session_state.uploaded_annotations[base_name] = ann_file
                    
                    # Check if corresponding image exists
                    if st.session_state.uploaded_images:
                        image_names_base = [img.name.replace('.png', '').replace('.jpg', '').replace('.jpeg', '') 
                                          for img in st.session_state.uploaded_images]
                        if base_name in image_names_base:
                            matched_count += 1
                
                st.success(f"✅ {len(uploaded_annotations)} annotations uploaded")
                if matched_count > 0:
                    st.info(f"🔗 {matched_count} matched with images")
                elif st.session_state.uploaded_images:
                    st.warning("⚠️ No matching image files found. Check filenames!")
        
        # Show upload instructions
        if not st.session_state.uploaded_images:
            st.info(
                "**� How to Upload:**\n\n"
                "1. **Drag & Drop**: Drag image files directly onto the upload area above\n"
                "2. **Browse**: Click 'Browse files' button to select from your computer\n"
                "3. **Multiple Selection**: Hold Ctrl (Windows) or Cmd (Mac) to select multiple files\n\n"
                "**Supported formats**: PNG, JPG, JPEG"
            )
            st.stop()
        
        st.markdown("---")
        
        # Show uploaded files summary
        with st.expander("📂 Uploaded Files Summary", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Images:**")
                for img in st.session_state.uploaded_images:
                    base_name = img.name.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
                    has_gt = base_name in st.session_state.uploaded_annotations
                    icon = "✅" if has_gt else "📸"
                    st.markdown(f"{icon} `{img.name}`")
            
            with col2:
                if st.session_state.uploaded_annotations:
                    st.markdown("**Ground Truth:**")
                    for base_name in st.session_state.uploaded_annotations:
                        st.markdown(f"📝 `{base_name}.txt`")
                else:
                    st.info("No ground truth files uploaded")
        
        all_images = st.session_state.uploaded_images
        image_names = [f.name for f in all_images]
    else:
        # Default dataset mode - Use TEST fractured images
        test_fractured = get_test_fractured_images()
        all_images = [img_path for idx, img_path in test_fractured]
        image_names = [f"{idx:04d}.png" for idx, img_path in test_fractured]
    
    if len(all_images) == 0:
        st.warning("No images found!")
        st.stop()
    
    st.sidebar.markdown("---")
    
    # Image selector
    selected_image_name = st.sidebar.selectbox(
        "Select Image",
        image_names,
        index=st.session_state.current_image_idx
    )
    
    # Update current index
    st.session_state.current_image_idx = image_names.index(selected_image_name)
    
    st.sidebar.markdown("---")
    
    # Display options
    st.sidebar.subheader("Display Options")
    show_rct = st.sidebar.checkbox("Show RCT Bboxes", value=True)
    show_gt = st.sidebar.checkbox("Show Ground Truth Lines", value=True)
    bbox_alpha = st.sidebar.slider("Bbox Transparency", 0.0, 1.0, 0.3, 0.05)
    
    st.sidebar.markdown("---")
    
    # Detection thresholds
    st.sidebar.subheader("Detection Thresholds")
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    iou_threshold = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05)
    
    st.sidebar.markdown("---")
    
    # Bbox expansion
    st.sidebar.subheader("🔍 Bbox Expansion")
    st.sidebar.markdown("_Expand bboxes to capture full RCT area_")
    scale_factor = st.sidebar.slider("Scale Factor", 1.0, 3.0, 2.0, 0.1, 
                                      help="2.0 = 2x larger bbox (recommended for Stage 2)")
    
    # Load image based on mode
    if mode == "📁 Upload Images":
        # Get selected uploaded file
        selected_file = all_images[st.session_state.current_image_idx]
        image = Image.open(selected_file)
        
        # Detect RCT
        with st.spinner("Detecting RCT teeth..."):
            # Save temp file for YOLO
            temp_path = Path("temp_upload.png")
            image.save(temp_path)
            detections = detect_rct(model, temp_path, conf_threshold, iou_threshold, scale_factor)
            temp_path.unlink()  # Delete temp file
        
        # Check if ground truth annotation exists for this image
        # Try multiple variations of the filename
        image_base_name = selected_file.name
        for ext in ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG']:
            image_base_name = image_base_name.replace(ext, '')
        
        if image_base_name in st.session_state.uploaded_annotations:
            annotation_file = st.session_state.uploaded_annotations[image_base_name]
            ground_truth_lines = load_ground_truth_from_uploaded_file(annotation_file)
        else:
            ground_truth_lines = []
        
    else:
        # Default dataset mode - TEST images
        test_fractured = get_test_fractured_images()
        selected_idx, image_path = test_fractured[st.session_state.current_image_idx]
        
        if not image_path.exists():
            st.error(f"Image not found: {image_path}")
            st.stop()
        
        image = Image.open(image_path)
        
        # Detect RCT
        with st.spinner("Detecting RCT teeth..."):
            detections = detect_rct(model, image_path, conf_threshold, iou_threshold, scale_factor)
        
        # Load ground truth
        ground_truth_lines = load_ground_truth_lines(selected_idx)
    
    # Show statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("RCT Detections", len(detections))
    with col2:
        st.metric("Ground Truth Lines", len(ground_truth_lines))
    with col3:
        avg_conf = np.mean([d['conf'] for d in detections]) if detections else 0.0
        st.metric("Avg Confidence", f"{avg_conf:.3f}")
    with col4:
        # Show if current image has ground truth
        if mode == "📁 Upload Images":
            has_gt = len(ground_truth_lines) > 0
            gt_status = "✅ Available" if has_gt else "❌ None"
            st.metric("Ground Truth", gt_status)
    
    # Draw visualization
    vis_image = draw_visualization(
        image.copy(),
        detections,
        ground_truth_lines,
        show_rct=show_rct,
        show_gt=show_gt,
        bbox_alpha=bbox_alpha
    )
    
    # Display
    st.image(vis_image, use_container_width=True)
    
    # Show details
    with st.expander("📊 Detection Details"):
        if detections:
            st.subheader("RCT Detections")
            for i, det in enumerate(detections):
                bbox = det['bbox']
                conf = det['conf']
                st.write(f"**Detection {i+1}**: Conf={conf:.3f}, Bbox=[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
        else:
            st.info("No RCT detections")
        
        st.markdown("---")
        
        if ground_truth_lines:
            st.subheader("Ground Truth Lines")
            for i, line in enumerate(ground_truth_lines):
                st.write(f"**Line {i+1}**: ({line[0]:.1f}, {line[1]:.1f}) → ({line[2]:.1f}, {line[3]:.1f})")
        else:
            st.info("No ground truth lines")
    
    # Navigation buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    current_idx = st.session_state.current_image_idx
    
    with col1:
        if current_idx > 0:
            if st.button("⬅️ Previous"):
                st.session_state.current_image_idx -= 1
                st.rerun()
    
    with col2:
        st.write(f"**Image {current_idx + 1} / {len(all_images)}**")
    
    with col3:
        if current_idx < len(all_images) - 1:
            if st.button("➡️ Next"):
                st.session_state.current_image_idx += 1
                st.rerun()
    
    # Overall statistics (only for default dataset mode)
    if mode == "💾 Use Default Dataset":
        test_images = get_test_fractured_images()
        test_split = load_test_split()
        st.sidebar.markdown("---")
        st.sidebar.subheader("📈 Test Dataset Statistics")
        st.sidebar.write(f"**Total Test Images:** {len(test_split)}")
        st.sidebar.write(f"**Test Fractured Images:** {len(test_images)}")
        st.sidebar.write(f"**Currently Viewing:** Image {current_idx + 1} / {len(test_images)}")
    
    # Legend
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎨 Legend")
    st.sidebar.markdown("🟢 **Green Box**: RCT Detection")
    st.sidebar.markdown("   - Transparent fill (adjustable)")
    st.sidebar.markdown("   - Confidence score on top")
    st.sidebar.markdown("🔴 **Red Line**: Ground Truth Fracture")
    st.sidebar.markdown("   - Thick line between points")
    st.sidebar.markdown("⚪ **White Circle**: Line Endpoints")
    
    # Instructions
    st.sidebar.markdown("---")
    st.sidebar.subheader("💡 Quick Guide")
    if mode == "📁 Upload Images":
        st.sidebar.info(
            "**Upload Mode:**\n\n"
            "🖱️ **Drag & Drop** files onto upload areas\n\n"
            "📸 **Images**: PNG/JPG files\n\n"
            "📝 **Ground Truth** (optional):\n"
            "- .txt files with same name as images\n"
            "- Format: `x1 y1` then `x2 y2` (two lines per fracture)\n\n"
            "🎚️ **Adjust**: Transparency, thresholds, visibility\n\n"
            "⬅️➡️ **Navigate**: Previous/Next buttons"
        )
    else:
        st.sidebar.info(
            "**Dataset Mode:**\n\n"
            "Browse the default fractured dataset with:\n"
            "- ✅ Automatic RCT detection\n"
            "- ✅ Pre-loaded ground truth\n"
            "- 📊 Overall statistics\n"
            "- ⬅️➡️ Easy navigation"
        )


if __name__ == "__main__":
    main()
