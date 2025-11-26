"""
Interactive Ground Truth Annotation Tool

Bu tool ile:
1. Görüntü açılır
2. Mouse ile kırık bölgeyi işaretlersiniz (bbox)
3. Ground truth bbox kaydedilir
4. Model prediction ile karşılaştırma yapılır

Usage:
    python annotate_ground_truth.py --image <path>
    
Instructions:
    - Sol tık: Bbox başlangıç noktası
    - Sürükle: Bbox çiz
    - 's' tuşu: Kaydet
    - 'r' tuşu: Sıfırla
    - 'q' tuşu: Çık

Author: Master's Thesis Project
Date: November 23, 2025
"""

import cv2
import numpy as np
import json
from pathlib import Path
import argparse


class GroundTruthAnnotator:
    """Interactive bbox annotation tool"""
    
    def __init__(self, image_path: str):
        self.image_path = image_path
        
        # Load image (handle Unicode)
        with open(image_path, 'rb') as f:
            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
        self.image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if self.image is None:
            raise ValueError(f"Failed to load: {image_path}")
        
        self.original = self.image.copy()
        self.bboxes = []
        
        # Drawing state
        self.drawing = False
        self.start_point = None
        self.current_bbox = None
        
        self.window_name = "Ground Truth Annotation"
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start drawing
            self.drawing = True
            self.start_point = (x, y)
            self.current_bbox = None
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # Update current bbox
                self.current_bbox = (
                    min(self.start_point[0], x),
                    min(self.start_point[1], y),
                    max(self.start_point[0], x),
                    max(self.start_point[1], y)
                )
        
        elif event == cv2.EVENT_LBUTTONUP:
            # Finish drawing
            if self.drawing and self.current_bbox:
                self.bboxes.append({
                    'bbox': self.current_bbox,
                    'label': 'fracture'
                })
                print(f"✓ Added bbox: {self.current_bbox}")
            
            self.drawing = False
            self.start_point = None
            self.current_bbox = None
    
    def draw_overlay(self):
        """Draw bboxes on image"""
        display = self.original.copy()
        
        # Draw saved bboxes
        for i, item in enumerate(self.bboxes):
            x1, y1, x2, y2 = item['bbox']
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(
                display,
                f"GT #{i+1}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
        
        # Draw current bbox (being drawn)
        if self.current_bbox:
            x1, y1, x2, y2 = self.current_bbox
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 0), 2)
        
        # Instructions
        h, w = display.shape[:2]
        overlay = display.copy()
        cv2.rectangle(overlay, (10, h - 120), (400, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
        
        instructions = [
            "Mouse: Draw bbox",
            "S: Save annotations",
            "R: Reset all",
            "Q: Quit"
        ]
        
        for i, text in enumerate(instructions):
            cv2.putText(
                display,
                text,
                (20, h - 95 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )
        
        return display
    
    def run(self):
        """Main annotation loop"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("="*80)
        print("GROUND TRUTH ANNOTATION")
        print("="*80)
        print(f"Image: {self.image_path}")
        print("\nInstructions:")
        print("  - Click and drag to draw bbox")
        print("  - Press 's' to save")
        print("  - Press 'r' to reset")
        print("  - Press 'q' to quit")
        print("="*80)
        
        while True:
            display = self.draw_overlay()
            cv2.imshow(self.window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                # Quit
                break
            
            elif key == ord('s'):
                # Save
                if self.bboxes:
                    self.save_annotations()
                else:
                    print("⚠ No bboxes to save!")
            
            elif key == ord('r'):
                # Reset
                self.bboxes = []
                print("↺ Reset all bboxes")
        
        cv2.destroyAllWindows()
    
    def save_annotations(self):
        """Save annotations to JSON"""
        output_dir = Path("outputs/ground_truth")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_name = Path(self.image_path).stem
        output_path = output_dir / f"{image_name}_gt.json"
        
        data = {
            'image': str(self.image_path),
            'image_size': {
                'width': self.original.shape[1],
                'height': self.original.shape[0]
            },
            'annotations': self.bboxes
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✓ Saved annotations to: {output_path}")
        print(f"  Total bboxes: {len(self.bboxes)}")


def main():
    parser = argparse.ArgumentParser(description="Ground Truth Annotation Tool")
    parser.add_argument('--image', type=str, required=True, help="Path to image")
    
    args = parser.parse_args()
    
    annotator = GroundTruthAnnotator(args.image)
    annotator.run()


if __name__ == "__main__":
    main()
