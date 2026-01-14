# =========================
# FIX PYTHON PATH
# =========================
import sys
import os
import glob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# =========================
# IMPORTS
# =========================
import torch
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import csv

# DATASETS
from datasets.nisqa_mel_dataset import NISQAMelDataset
from datasets.nisqa_mfcc_dataset import NISQAMFCCDataset
from datasets.nisqa_wav2vec_dataset import NISQAWav2VecDataset

# MODELS
from models.nisqa_cnn import NISQACNN
from models.nisqa_cnn_gru import NISQACNN_GRU
from models.nisqa_cnn_lstm import NISQACNN_LSTM
from models.nisqa_wav2vec_rnn import NISQAWav2VecRNN

# =========================
# CONFIG
# =========================
ROOT_DIR = r"C:\Users\Danyill\PycharmProjects\DyplomNSQC\NISQA_Corpus"
CSV_FILES = glob.glob(os.path.join(ROOT_DIR, "**", "*_file.csv"), recursive=True)

TRAIN_CSVS = [f for f in CSV_FILES if "TRAIN" in f.upper()]
VAL_CSVS   = [f for f in CSV_FILES if "VAL" in f.upper()]

BATCH_SIZE = 8
EPOCHS = 50
LR = 3e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {DEVICE}")

# =========================
# FEATURE SELECTION
# =========================
print("\nSelect input features:")
print("1 - Mel-spectrogram")
print("2 - MFCC")
print("3 - wav2vec2")

feat_choice = input("Choice [1/2/3]: ").strip()

def build_dataset(csvs):
    datasets = []
    for csv_path in csvs:
        if feat_choice == "1":
            datasets.append(NISQAMelDataset(csv_path, ROOT_DIR, normalize_mos=True))
        elif feat_choice == "2":
            datasets.append(NISQAMFCCDataset(csv_path, ROOT_DIR))
        elif feat_choice == "3":
            datasets.append(NISQAWav2VecDataset(csv_path, ROOT_DIR))
    return ConcatDataset(datasets)

if feat_choice == "1":
    FEATURE_NAME = "mel"
    feature_type = "cnn"
elif feat_choice == "2":
    FEATURE_NAME = "mfcc"
    feature_type = "cnn"
elif feat_choice == "3":
    FEATURE_NAME = "wav2vec2"
    feature_type = "wav2vec"
else:
    raise ValueError("Invalid feature choice")

train_dataset = build_dataset(TRAIN_CSVS)
val_dataset   = build_dataset(VAL_CSVS)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =========================
# MODEL SELECTION
# =========================
print("\nSelect model:")

if feature_type == "cnn":
    print("1 - CNN")
    print("2 - CNN + GRU")
    print("3 - CNN + LSTM")
else:
    print("1 - wav2vec2 + GRU")
    print("2 - wav2vec2 + LSTM")

choice = input("Choice: ").strip()

if feature_type == "cnn":
    if choice == "1":
        MODEL_NAME = "CNN"
        model = NISQACNN()
    elif choice == "2":
        MODEL_NAME = "CNN_GRU"
        model = NISQACNN_GRU()
    elif choice == "3":
        MODEL_NAME = "CNN_LSTM"
        model = NISQACNN_LSTM()
    else:
        raise ValueError("Invalid model choice")
else:
    if choice == "1":
        MODEL_NAME = "WAV2VEC_GRU"
        model = NISQAWav2VecRNN(rnn_type="gru")
    elif choice == "2":
        MODEL_NAME = "WAV2VEC_LSTM"
        model = NISQAWav2VecRNN(rnn_type="lstm")
    else:
        raise ValueError("Invalid model choice")

model = model.to(DEVICE)
print(f"[INFO] Selected model: {MODEL_NAME}")

# =========================
# OUTPUT DIR
# =========================
OUT_DIR = os.path.join("outputs", f"{MODEL_NAME}_{FEATURE_NAME}")
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# TRAINING SETUP
# =========================
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

def rmse(y_pred, y_true):
    return np.sqrt(np.mean((y_pred - y_true) ** 2))

train_mse_hist, val_mse_hist = [], []
rmse_hist, pearson_hist, spearman_hist = [], [], []

best_val_mse = float("inf")
best_epoch = -1
best_metrics = {}
# =========================
# TRAINING LOOP
# =========================
for epoch in range(EPOCHS):
    print(f"\n===== Epoch {epoch+1}/{EPOCHS} =====")

    model.train()
    train_losses = []

    for x, mos in tqdm(train_loader, desc="Train"):
        x, mos = x.to(DEVICE), mos.to(DEVICE)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, mos)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    model.eval()
    val_losses = []
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, mos in tqdm(val_loader, desc="Val"):
            x, mos = x.to(DEVICE), mos.to(DEVICE)
            pred = model(x)
            val_losses.append(criterion(pred, mos).item())
            y_true.append(mos.cpu().numpy())
            y_pred.append(pred.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    if epoch == EPOCHS - 1:
        y_true_last = y_true
        y_pred_last = y_pred

    train_mse = np.mean(train_losses)
    val_mse   = np.mean(val_losses)
    val_rmse  = rmse(y_pred, y_true)
    pearson   = pearsonr(y_true, y_pred)[0]
    spearman  = spearmanr(y_true, y_pred)[0]

    train_mse_hist.append(train_mse)
    val_mse_hist.append(val_mse)
    rmse_hist.append(val_rmse)
    pearson_hist.append(pearson)
    spearman_hist.append(spearman)

    print(f"Train MSE: {train_mse:.4f}")
    print(f"Val   MSE: {val_mse:.4f}")
    print(f"RMSE:      {val_rmse:.4f}")
    print(f"Pearson:   {pearson:.4f}")
    print(f"Spearman:  {spearman:.4f}")

    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_epoch = epoch + 1
        best_metrics = {
            "val_mse": val_mse,
            "rmse": val_rmse,
            "pearson": pearson,
            "spearman": spearman
        }


# =========================
# GRAPHS
# =========================
epochs = range(1, EPOCHS + 1)

plt.figure()
plt.plot(epochs, train_mse_hist, label="Train MSE")
plt.plot(epochs, val_mse_hist, label="Validation MSE")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title(f"MSE Curves\n{MODEL_NAME} ({FEATURE_NAME})")
plt.legend()
plt.grid()
plt.savefig(os.path.join(OUT_DIR, "mse.png"))
plt.close()

plt.figure()
plt.plot(epochs, rmse_hist, label="RMSE")
plt.plot(epochs, pearson_hist, label="Pearson")
plt.plot(epochs, spearman_hist, label="Spearman")
plt.xlabel("Epoch")
plt.ylabel("Metric value")
plt.title(f"Validation Metrics\n{MODEL_NAME} ({FEATURE_NAME})")
plt.legend()
plt.grid()
plt.savefig(os.path.join(OUT_DIR, "metrics.png"))
plt.close()

# ---- MOS 0–1 ----
plt.figure()
plt.scatter(y_true_last, y_pred_last, alpha=0.5)
plt.plot([0, 1], [0, 1], "r--")
plt.xlabel("True MOS (0–1)")
plt.ylabel("Predicted MOS (0–1)")
plt.title(f"MOS Prediction (Normalized)\n{MODEL_NAME} ({FEATURE_NAME})")
plt.grid()
plt.savefig(os.path.join(OUT_DIR, "mos_0_1.png"))
plt.close()

# ---- MOS 1–5 ----
y_true_5 = np.clip(y_true_last * 4 + 1, 1, 5)
y_pred_5 = np.clip(y_pred_last * 4 + 1, 1, 5)

plt.figure()
plt.scatter(y_true_5, y_pred_5, alpha=0.5)
plt.plot([1, 5], [1, 5], "r--")
plt.xlabel("True MOS (1–5)")
plt.ylabel("Predicted MOS (1–5)")
plt.title(f"MOS Prediction (1–5)\n{MODEL_NAME} ({FEATURE_NAME})")
plt.grid()
plt.savefig(os.path.join(OUT_DIR, "mos_1_5.png"))
plt.close()

# ---- ERROR ----
errors = y_pred_5 - y_true_5
plt.figure()
plt.hist(errors, bins=30)
plt.xlabel("Prediction Error (MOS)")
plt.ylabel("Samples")
plt.title(f"Error Distribution\n{MODEL_NAME} ({FEATURE_NAME})")
plt.grid()
plt.savefig(os.path.join(OUT_DIR, "error.png"))
plt.close()

torch.save(model.state_dict(), os.path.join(OUT_DIR, "model.pt"))
print("[INFO] Training finished")


# =========================
# SAVE BEST METRICS TO CSV
# =========================
csv_path = "outputs/model_comparison.csv"
file_exists = os.path.isfile(csv_path)

with open(csv_path, mode="a", newline="") as f:
    writer = csv.writer(f)

    if not file_exists:
        writer.writerow([
            "Model",
            "Features",
            "Best_Epoch",
            "Val_MSE",
            "RMSE",
            "Pearson",
            "Spearman"
        ])

    writer.writerow([
        MODEL_NAME,
        FEATURE_NAME,
        best_epoch,
        f"{best_metrics['val_mse']:.4f}",
        f"{best_metrics['rmse']:.4f}",
        f"{best_metrics['pearson']:.4f}",
        f"{best_metrics['spearman']:.4f}"
    ])

print(f"[INFO] Best results saved to {csv_path}")