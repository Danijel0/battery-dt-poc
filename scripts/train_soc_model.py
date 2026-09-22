"""
CNN-LSTM SoC Model Training (no MLflow)
"""

import os
import json
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

DATA_PATH      = "data/soc_training_data.h5"
MODEL_DIR      = "data/models"
WINDOW_SIZE    = 60
N_FEATURES     = 4
BATCH_SIZE     = 2048    # increased from 256: fewer batches/epoch, faster CPU training
EPOCHS         = 30
LR             = 1e-3
TRAIN_RATIO    = 0.8
SUBSAMPLE_FRAC = 0.20    # stratified subsample: 20% balances training speed with
                         # sufficient data variety for <2% RMSE target
SEED           = 42

os.makedirs(MODEL_DIR, exist_ok=True)
torch.manual_seed(SEED)


class CNNLSTMSoC(nn.Module):
    def __init__(self, n_features=4, cnn_filters=64, kernel_size=3,
                 lstm_hidden=128, lstm_layers=1, dropout=0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, cnn_filters, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(cnn_filters, cnn_filters, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(cnn_filters, lstm_hidden, lstm_layers,
                            batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1]).squeeze(1)


def train():
    # GPU detection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")   # Apple Silicon
        print("GPU: Apple MPS")
    else:
        device = torch.device("cpu")
        print("GPU: not available, using CPU")

    with h5py.File(DATA_PATH, "r") as f:
        X_np = f["X"][:]
        y_np = f["y"][:]
    print(f"Loaded: X={X_np.shape}  y={y_np.shape}")

    # Stratified subsample: bin SoC into 20 buckets, sample uniformly from each.
    # Sliding windows from same discharge are highly correlated — subsampling
    # preserves the full SoC distribution while reducing training time ~3x.
    rng      = np.random.default_rng(SEED)
    n_bins   = 20
    bins     = np.linspace(0, 1, n_bins + 1)
    bin_idx  = np.digitize(y_np, bins) - 1
    keep     = []
    for b in range(n_bins):
        idx = np.where(bin_idx == b)[0]
        n_keep = max(1, int(len(idx) * SUBSAMPLE_FRAC))
        keep.extend(rng.choice(idx, size=n_keep, replace=False).tolist())
    keep = np.array(keep)
    rng.shuffle(keep)
    X_np = X_np[keep]
    y_np = y_np[keep]
    print(f"Subsampled: {len(keep):,} samples ({SUBSAMPLE_FRAC*100:.0f}%, stratified by SoC)")

    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)

    dataset = TensorDataset(X, y)
    n_train = int(len(dataset) * TRAIN_RATIO)
    n_val   = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(SEED))

    # num_workers=2 speeds up data loading on CPU; pin_memory helps GPU transfer
    n_workers    = 2 if device.type == "cpu" else 4
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=n_workers, pin_memory=device.type=="cuda")
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=n_workers, pin_memory=device.type=="cuda")
    model     = CNNLSTMSoC().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.MSELoss()

    best_val_rmse   = float("inf")
    best_model_path = os.path.join(MODEL_DIR, "cnn_lstm_soc_best.pt")

    print(f"Training {EPOCHS} epochs — {n_train} train / {n_val} val samples\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(X_b)
            loss = criterion(pred, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                val_preds.extend(model(X_b).cpu().numpy())
                val_targets.extend(y_b.cpu().numpy())

        train_rmse = float(np.sqrt(np.mean(train_losses)))
        val_rmse   = float(np.sqrt(np.mean((np.array(val_preds) - np.array(val_targets))**2)))
        val_mae    = float(np.mean(np.abs(np.array(val_preds) - np.array(val_targets))))

        scheduler.step(val_rmse)
        print(f"Epoch {epoch:02d}/{EPOCHS}  train_rmse={train_rmse:.4f}  "
              f"val_rmse={val_rmse:.4f}  val_mae={val_mae:.4f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), best_model_path)

    print(f"\nBest val RMSE : {best_val_rmse:.4f}  ({best_val_rmse*100:.2f}%)")
    print(f"Target        : 0.0200  (2.00%)")
    print(f"Model saved   : {best_model_path}")

    with open(os.path.join(MODEL_DIR, "soc_model_info.json"), "w") as f:
        json.dump({
            "model_class":   "CNNLSTMSoC",
            "window_size":   WINDOW_SIZE,
            "n_features":    N_FEATURES,
            "best_val_rmse": best_val_rmse,
            "model_path":    best_model_path,
        }, f, indent=2)


if __name__ == "__main__":
    train()
