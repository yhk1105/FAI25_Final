import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train supervised policy network (5 actions).")
    parser.add_argument(
        "--data",
        default="data/poker_data_with_baseline_final.npz",
        help="Path to training npz (default: data/poker_data_with_baseline_final.npz)",
    )
    parser.add_argument(
        "--out",
        default="models/policy_nn.pt",
        help="Output path for model weights (default: models/policy_nn.pt)",
    )
    return parser.parse_args()


args = parse_args()

# 讀檔案（改成相對路徑，方便丟上 GitHub）
data = np.load(args.data)
print(data)
X = data["states"]
y = data["actions"]

print(f"X shape: {X.shape}, y shape: {y.shape}")

# 劃分訓練/驗證集
split = int(0.9 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# 轉成 torch tensor
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_val = torch.FloatTensor(X_val)
y_val = torch.LongTensor(y_val)

# DataLoader
batch_size = 1024
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# 模型架構


class PokerPolicyNet(nn.Module):
    def __init__(self, input_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)

model = PokerPolicyNet(input_dim=X.shape[1], n_actions=5).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

n_epochs = 10000
patience = 20           # 連續多少 epoch 沒有改善就 early stop
best_val_loss = float("inf")
patience_counter = 0

train_losses, val_losses = [], []
best_epoch = 0
for epoch in tqdm(range(n_epochs)):
    model.train()
    running_loss = 0
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    train_loss = running_loss / len(train_loader.dataset)

    # 驗證
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            running_loss += loss.item() * xb.size(0)
            predicted = pred.argmax(dim=1)
            correct += (predicted == yb).sum().item()
            total += yb.size(0)
    val_loss = running_loss / len(val_loader.dataset)
    val_acc = correct / total

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Early Stopping 機制
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_epoch = epoch
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# === 重新訓練：train + val 合併 ===
X_all = torch.cat([X_train, X_val], dim=0)
y_all = torch.cat([y_train, y_val], dim=0)
all_dataset = TensorDataset(X_all, y_all)
all_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=True)

model = PokerPolicyNet(input_dim=X.shape[1], n_actions=5).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for i in range(best_epoch+1):
    model.train()
    running_loss = 0
    for xb, yb in all_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    train_loss = running_loss / len(all_loader.dataset)
    print(f"[Full Train] Epoch {i+1}/{best_epoch+1} | Loss: {train_loss:.4f}")

# 儲存模型
torch.save(model.state_dict(), args.out)
print(f"Best model saved as {args.out}")

# 畫圖
plt.plot(train_losses, label="Train loss")
plt.plot(val_losses, label="Val loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Poker Policy NN Loss Curve")
plt.show()
