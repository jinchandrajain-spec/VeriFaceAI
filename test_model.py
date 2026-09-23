import os
import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

# Restore the exact CustomFaceCNN architecture from training
class CustomFaceCNN(nn.Module):
    def __init__(self):
        super(CustomFaceCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 128), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(128, 1), nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def load_model(model_path, device):
    """Safely loads the model weights into the CustomFaceCNN architecture."""
    if not os.path.exists(model_path):
        print(f"\n[ERROR] Model weights not found at '{model_path}'.")
        print("Please ensure 'train_model.py' has finished running and produced 'best_model.pth'.\n")
        return None
        
    model = CustomFaceCNN()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval() # Ensure model is strictly set to evaluation mode
    return model

def predict_single_image(model, image_path, device):
    """CLI test mode for a single image."""
    if not os.path.exists(image_path):
        print(f"\n[ERROR] Target image not found at '{image_path}'.\n")
        return
        
    # Apply standard val/test transforms. 
    # Force resize just in case the provided image isn't natively 128x128
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor).item()
        
    # PyTorch ImageFolder assigns classes alphabetically (FAKE=0, REAL=1)
    if output > 0.5:
        pred_label = "REAL (Authentic)"
        confidence = output * 100
    else:
        pred_label = "FAKE (Manipulated)"
        confidence = (1 - output) * 100
        
    print(f"\n" + "="*40)
    print(f"{'SINGLE IMAGE PREDICTION':^40}")
    print("="*40)
    print(f"Target Image : {image_path}")
    print(f"Prediction   : {pred_label}")
    print(f"Confidence   : {confidence:.2f}%")
    print(f"Raw Output   : {output:.4f}")
    print("="*40)
    
    # Save an annotated copy of the image
    annotated_img = img.copy()
    draw = ImageDraw.Draw(annotated_img)
    text = f"{pred_label} - {confidence:.1f}%"
    
    # Add crude shadow text to ensure it's visible on most backgrounds
    draw.text((6, 6), text, fill="black")
    draw.text((5, 5), text, fill="red")
    
    output_path = "annotated_prediction.png"
    annotated_img.save(output_path)
    print(f"\nSuccessfully saved annotated prediction to '{output_path}'.")


def evaluate_dataset(model, device):
    """Evaluates the entire test dataset and graphs confusion metrics & demographics."""
    test_dir = "dataset/test"
    if not os.path.exists(test_dir):
        print(f"\n[ERROR] Test dataset folder not found at '{test_dir}'.\n")
        return
        
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}
    
    all_preds = []
    all_labels = []
    all_paths = []
    
    print("\nRunning test evaluation... (No Gradients)")
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(test_loader, desc="Testing Model")):
            images = images.to(device)
            outputs = model(images).squeeze()
            
            # Handle batch sizes of 1
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
                
            preds = (outputs > 0.5).int().cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            
            start_idx = i * 32
            end_idx = start_idx + len(images)
            batch_paths = [test_dataset.samples[idx][0] for idx in range(start_idx, min(end_idx, len(test_dataset.samples)))]
            all_paths.extend(batch_paths)
            
    # Core Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    print("\n" + "="*45)
    print(f"{'OVERALL TEST PERFORMANCE':^45}")
    print("="*45)
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f} (Real vs. Fake)")
    print(f"Recall    : {recall:.4f} (Real vs. Fake)")
    print(f"F1-Score  : {f1:.4f}")
    print("-" * 45)
    
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        print(f"True Positives  (REAL correctly classified): {tp}")
        print(f"True Negatives  (FAKE correctly classified): {tn}")
        print(f"False Positives (FAKE mistaken for REAL)   : {fp}")
        print(f"False Negatives (REAL mistaken for FAKE)   : {fn}")
    print("="*45)
    
    # Plot Visual Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[idx_to_class.get(0, 'FAKE'), idx_to_class.get(1, 'REAL')],
                yticklabels=[idx_to_class.get(0, 'FAKE'), idx_to_class.get(1, 'REAL')])
    plt.title('Confusion Matrix on Test Set')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('confusion_matrix.png', bbox_inches='tight')
    plt.close()
    print("\n[+] Saved confusion matrix visual to 'confusion_matrix.png'.")
    
    # Process Demographic Testing
    evaluate_demographics(all_paths, all_labels, all_preds)


def evaluate_demographics(paths, labels, preds):
    """Calculates evaluation metrics across specific dataset attributes mapping back to CSV."""
    csv_file = 'AI_Classification-Project.csv'
    if not os.path.exists(csv_file):
        fallback = 'AI_Classification-Project copy.csv'
        if os.path.exists(fallback):
            csv_file = fallback
        else:
            print("\n[WARNING] Metadata CSV not found. Skipping Demographic Sub-analysis.")
            return
            
    df = pd.read_csv(csv_file)
    
    # Extract unique numerical image_ids from paths like 'dataset/test/REAL/123.jpg'
    image_ids = [int(os.path.basename(p).split('.')[0]) for p in paths]
    
    results_df = pd.DataFrame({
        'image_id': image_ids,
        'true_label': labels,
        'pred_label': preds,
        'correct': [1 if t == p else 0 for t, p in zip(labels, preds)]
    })
    
    merged = pd.merge(results_df, df, on='image_id', how='left')
    
    attributes = ['gender', 'age_group', 'detection_difficulty']
    
    # Filter only to attributes that actually exist in the CSV columns
    attributes = [a for a in attributes if a in merged.columns]
    
    if not attributes:
        return
        
    fig, axes = plt.subplots(1, len(attributes), figsize=(6 * len(attributes), 5))
    if len(attributes) == 1:
        axes = [axes] # standardize to list
        
    fig.suptitle('Accuracy Breakdown by Demographics & Attributes', fontsize=16)
    
    print("\n" + "="*45)
    print(f"{'DEMOGRAPHIC & ATTRIBUTE ACCURACY':^45}")
    print("="*45)
    
    for i, attr in enumerate(attributes):
        group_acc = merged.groupby(attr)['correct'].mean().reset_index()
        group_acc['correct'] *= 100 # convert to percentage
        
        print(f"\n--- {attr.replace('_', ' ').title()} ---")
        for _, row in group_acc.iterrows():
            print(f"{row[attr]:>10} : {row['correct']:.1f}%")
            
        sns.barplot(data=group_acc, x=attr, y='correct', ax=axes[i], palette='viridis', hue=attr, legend=False)
        axes[i].set_title(f'Accuracy by {attr.replace("_", " ").title()}')
        axes[i].set_ylabel('Accuracy (%)')
        axes[i].set_ylim(0, 105)
        
    plt.tight_layout()
    plt.savefig('demographic_test_results.png', bbox_inches='tight')
    plt.close()
    print("\n[+] Saved demographic attribute charts to 'demographic_test_results.png'.")
    print("="*45)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Custom Face CNN Model")
    parser.add_argument('--image', type=str, help="Absolute or relative path to a single image for quick prediction")
    args = parser.parse_args()
    
    device = get_device()
    print(f"Initializing Testing Suite on device: {device}...")
    
    model = load_model("best_model.pth", device)
    if model is None:
        return
        
    if args.image:
        predict_single_image(model, args.image, device)
    else:
        evaluate_dataset(model, device)

if __name__ == "__main__":
    main()
