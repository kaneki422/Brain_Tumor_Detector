import os, glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from preprocessing import preprocess_image
from feature_extraction import get_feature_vector
from traditional_ml import train_all
from cnn_model import BrainTumorCNN

DATA_DIR = 'data/'      
def load_dataset():
    X_feat, X_img, y = [], [], []
    for label, folder in [(1, 'yes'), (0, 'no')]:
        for path in glob.glob(f'{DATA_DIR}{folder}/*.jpg'):
            norm, raw = preprocess_image(path)
            X_feat.append(get_feature_vector(norm))
            X_img.append(norm)
            y.append(label)
    return (np.array(X_feat), np.array(X_img), np.array(y))

def train_cnn(X_img, y, epochs=20, save_path='models/cnn.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Shape: (N, 1, 128, 128)
    X_t = torch.FloatTensor(X_img).unsqueeze(1)
    y_t = torch.LongTensor(y)

    ds = TensorDataset(X_t, y_t)
    train_size = int(0.8 * len(ds))
    train_ds, val_ds = random_split(ds, [train_size, len(ds) - train_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=32)

    model = BrainTumorCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    history = {'train_acc': [], 'val_acc': []}
    for epoch in range(epochs):
        model.train()
        correct = total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            correct += (out.argmax(1) == yb).sum().item()
            total   += len(yb)
        train_acc = correct / total

        model.eval()
        vc = vt = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                vc += (model(xb).argmax(1) == yb).sum().item()
                vt += len(yb)
        val_acc = vc / vt
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        print(f"Epoch {epoch+1}/{epochs} | Train: {train_acc:.3f} | Val: {val_acc:.3f}")

    torch.save(model.state_dict(), save_path)
    return history

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    X_feat, X_img, y = load_dataset()
    print("Training Traditional ML models...")
    ml_results = train_all(X_feat, y)
    print("\nTraining CNN...")
    cnn_history = train_cnn(X_img, y)