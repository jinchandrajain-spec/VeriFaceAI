import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

class CustomFaceCNN(nn.Module):
    def __init__(self):
        super(CustomFaceCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x

def main():
    # Detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Data Pipeline
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    test_val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    batch_size = 32
    
    # Load Datasets using ImageFolder
    try:
        train_dataset = datasets.ImageFolder(root="dataset/train", transform=train_transforms)
        val_dataset = datasets.ImageFolder(root="dataset/val", transform=test_val_transforms)
        test_dataset = datasets.ImageFolder(root="dataset/test", transform=test_val_transforms)
    except FileNotFoundError as e:
        print(f"Error loading datasets. Ensure the 'dataset/' directory exists and contains images. Details: {e}")
        return

    print(f"Class mapping: {train_dataset.class_to_idx}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize Model, Loss, and Optimizer
    model = CustomFaceCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    epochs = 20
    best_val_loss = float('inf')
    
    # Training & Validation Loop
    print("\n--- Starting Training ---")
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
        train_loss /= len(train_dataset)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images = images.to(device)
                labels = labels.to(device).float().unsqueeze(1)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                
                preds = (outputs > 0.5).float()
                val_correct += (preds == labels).sum().item()
                
        val_loss /= len(val_dataset)
        val_acc = val_correct / len(val_dataset)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"-> Saved new best model with Val Loss: {best_val_loss:.4f}")
            
    print("\n--- Training Complete ---")
    
    # Test Evaluation
    print("\n--- Starting Test Evaluation ---")
    # Load the best model weights
    model.load_state_dict(torch.load("best_model.pth", map_location=device, weights_only=True))
    model.eval()
    
    test_probs = []
    test_preds = []
    test_trues = []
    test_paths = []
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(test_loader, desc="Test Evaluation")):
            images = images.to(device)
            outputs = model(images)
            
            probs = outputs.cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            
            test_probs.extend(probs)
            test_preds.extend(preds)
            test_trues.extend(labels.numpy())
            
            # Extract image paths corresponding to the current batch
            start_idx = i * batch_size
            end_idx = start_idx + len(images)
            batch_paths = [test_dataset.samples[idx][0] for idx in range(start_idx, min(end_idx, len(test_dataset.samples)))]
            test_paths.extend(batch_paths)
            
    # Compute Metrics
    acc = accuracy_score(test_trues, test_preds)
    precision = precision_score(test_trues, test_preds, zero_division=0)
    recall = recall_score(test_trues, test_preds, zero_division=0)
    f1 = f1_score(test_trues, test_preds, zero_division=0)
    
    print("\n--- Test Metrics ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Save predictions to CSV
    idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}
    true_class = [idx_to_class[label] for label in test_trues]
    pred_class = [idx_to_class[pred] for pred in test_preds]
    
    df_preds = pd.DataFrame({
        'image_path': test_paths,
        'true_label': test_trues,
        'true_class': true_class,
        'pred_prob': test_probs,
        'pred_label': test_preds,
        'pred_class': pred_class
    })
    
    df_preds.to_csv("test_predictions.csv", index=False)
    print("\nSaved predictions to 'test_predictions.csv'.")

if __name__ == "__main__":
    main()
