import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from train_u2net_sr_clahe import U2Net_Lite, SegmentationDataset

def eval_u2net():
    dataset_dir = Path("auto_labeled_segmentation_sr_clahe")
    all_images = list((dataset_dir / "images").glob("*.png"))
    
    # Reproduce exact val split from training
    _, val_imgs = train_test_split(all_images, test_size=0.2, random_state=42)
    
    val_ds = SegmentationDataset(val_imgs, dataset_dir / "masks")
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = U2Net_Lite().to(device)
    
    weights_path = "runs/u2net_segmentation/best_u2net.pth"
    if not os.path.exists(weights_path):
        print(f"Error: {weights_path} not found!")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    pos_iou = []
    pos_dice = []
    precision_list = []
    recall_list = []
    
    healthy_acc = []

    print(f"\nU2-Net Lite loaded. Evaluating on {len(val_imgs)} validation images...")
    
    vis_samples = []

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            
            # Predict
            outputs = model(imgs)
            preds = torch.sigmoid(outputs)
            preds_bin = (preds > 0.5).float()
            
            for j in range(imgs.size(0)):
                p = preds_bin[j]
                t = masks[j]
                
                tp = (p * t).sum().item()
                fp = (p * (1 - t)).sum().item()
                fn = ((1 - p) * t).sum().item()
                tn = ((1 - p) * (1 - t)).sum().item()
                
                if t.sum().item() > 0: # Fractured
                    iou = tp / (tp + fp + fn + 1e-6)
                    dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)
                    prec = tp / (tp + fp + 1e-6)
                    rec = tp / (tp + fn + 1e-6)
                    
                    pos_iou.append(iou)
                    pos_dice.append(dice)
                    precision_list.append(prec)
                    recall_list.append(rec)
                    
                    if len(vis_samples) < 8:
                        vis_samples.append((imgs[j].cpu(), t.cpu(), preds[j].cpu(), preds_bin[j].cpu()))
                else: # Healthy
                    if fp < 50:
                        healthy_acc.append(1.0)
                    else:
                        healthy_acc.append(0.0)

    # Average metrics
    print("\n" + "="*50)
    print(f"📊 FRACTURED CROPS METRICS ({len(pos_iou)} images)")
    print("="*50)
    print(f"Mean IoU (Intersection over Union) : {np.mean(pos_iou):.4f}")
    print(f"Mean Dice Coefficient              : {np.mean(pos_dice):.4f}")
    print(f"Mean Precision (Are lines real?)   : {np.mean(precision_list):.4f}")
    print(f"Mean Recall (Did we catch lines?)  : {np.mean(recall_list):.4f}")
    
    print("\n" + "="*50)
    print(f"🛡️ HEALTHY CROPS CAPABILITIES ({len(healthy_acc)} images)")
    print("="*50)
    print(f"True Negative Accuracy (No False Alarms): {np.mean(healthy_acc)*100:.2f}%")
    
    # Plotting
    os.makedirs("outputs/u2net_eval", exist_ok=True)
    plt.figure(figsize=(16, 4 * len(vis_samples)))
    
    for idx, (img, gt, pred, pred_bin) in enumerate(vis_samples):
        # Unnormalize img
        img = img.permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        plt.subplot(len(vis_samples), 4, idx*4 + 1)
        plt.imshow(img)
        plt.title("Original Input")
        plt.axis('off')
        
        plt.subplot(len(vis_samples), 4, idx*4 + 2)
        plt.imshow(gt[0], cmap='gray')
        plt.title("Ground Truth Mask")
        plt.axis('off')
        
        plt.subplot(len(vis_samples), 4, idx*4 + 3)
        plt.imshow(pred[0], cmap='jet')
        plt.title("U2-Net Probability Map")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
        
        plt.subplot(len(vis_samples), 4, idx*4 + 4)
        plt.imshow(pred_bin[0], cmap='gray')
        plt.title("Final Definite Output (>0.5)")
        plt.axis('off')
        
    plt.tight_layout()
    save_path = "outputs/u2net_eval/visual_results_u2net.png"
    plt.savefig(save_path, dpi=150)
    print(f"\n🖼️ Visually compared actual predictions saved to:\n ---> {save_path}")

if __name__ == "__main__":
    eval_u2net()