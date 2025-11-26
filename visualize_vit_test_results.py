"""
Visualize ViT Binary Classifier Test Results
Displays all test images with their predictions and confidence scores
"""

import json
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import timm
from torchvision import transforms

def load_model_and_results(model_path, results_path):
    """Load the trained model and test results"""
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model architecture (same as training)
    backbone = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=0)
    hidden_dim = backbone.num_features
    
    class FractureBinaryClassifier(torch.nn.Module):
        def __init__(self, backbone, hidden_dim, num_classes=2, dropout=0.3):
            super().__init__()
            self.backbone = backbone
            self.classifier = torch.nn.Sequential(
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, 256),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(256, num_classes)
            )
        
        def forward(self, x):
            features = self.backbone(x)
            return self.classifier(features)
    
    model = FractureBinaryClassifier(backbone, hidden_dim)
    
    # Load checkpoint (contains epoch, optimizer state, etc.)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model, results, device

def get_prediction_with_confidence(model, image_path, device):
    """Get prediction and confidence score for an image"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return predicted.item(), confidence.item(), probabilities[0].cpu().numpy()

def visualize_test_results(model_path='runs/vit_classifier/best_model.pt',
                           results_path='runs/vit_classifier/results.json',
                           save_path='runs/vit_classifier/test_visualization.png'):
    """Visualize all test results in a grid"""
    
    print("Loading model and results...")
    model, results, device = load_model_and_results(model_path, results_path)
    
    test_predictions = results['test_predictions']
    num_images = len(test_predictions)
    
    # Calculate grid size
    cols = 5
    rows = (num_images + cols - 1) // cols
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    print(f"\nProcessing {num_images} test images...")
    
    # Class names
    class_names = ['No Fracture', 'Fracture']
    
    for idx, pred_info in enumerate(test_predictions):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        # Load image
        image_path = pred_info['image']
        true_label = pred_info['true_label']
        pred_label = pred_info['predicted_label']
        
        # Get confidence scores
        _, confidence, probs = get_prediction_with_confidence(model, image_path, device)
        
        # Load and display image
        img = Image.open(image_path).convert('RGB')
        ax.imshow(img)
        ax.axis('off')
        
        # Determine if prediction is correct
        is_correct = (true_label == pred_label)
        border_color = 'green' if is_correct else 'red'
        
        # Create title with prediction info
        true_class = class_names[true_label]
        pred_class = class_names[pred_label]
        
        title = f"True: {true_class}\n"
        title += f"Pred: {pred_class} ({confidence*100:.1f}%)\n"
        title += f"Prob: No={probs[0]*100:.1f}% | Yes={probs[1]*100:.1f}%"
        
        # Set title color based on correctness
        ax.set_title(title, fontsize=9, color=border_color, fontweight='bold')
        
        # Add border
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)
        
        # Print info
        status = "✓" if is_correct else "✗"
        print(f"{status} Image {idx+1}/{num_images}: {Path(image_path).name} - "
              f"True: {true_class}, Pred: {pred_class} ({confidence*100:.1f}%)")
    
    # Hide empty subplots
    for idx in range(num_images, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    # Add overall title
    test_metrics = results['test_metrics']
    fig.suptitle(
        f"ViT Binary Classifier - Test Results\n"
        f"Accuracy: {test_metrics['accuracy']*100:.1f}% | "
        f"Precision: {test_metrics['precision']*100:.1f}% | "
        f"Recall: {test_metrics['recall']*100:.1f}% | "
        f"F1: {test_metrics['f1_score']*100:.1f}%",
        fontsize=16, fontweight='bold', y=0.98
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {save_path}")
    print(f"\nTest Metrics:")
    print(f"  Accuracy:  {test_metrics['accuracy']*100:.2f}%")
    print(f"  Precision: {test_metrics['precision']*100:.2f}%")
    print(f"  Recall:    {test_metrics['recall']*100:.2f}%")
    print(f"  F1 Score:  {test_metrics['f1_score']*100:.2f}%")
    
    # Show plot
    plt.show()
    
    return fig

if __name__ == '__main__':
    visualize_test_results()
