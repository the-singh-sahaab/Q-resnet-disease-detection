#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
 Hybrid Quantum ResNet (QResNet) - Comprehensive Model Evaluation
============================================================================
 This script loads the saved QResNet checkpoint and performs an exhaustive
 performance evaluation:
   1. Model layer structure (full architecture printout + parameter count)
   2. All performance metrics (Accuracy, Precision, Recall, F1, F2, MCC,
      Cohen's Kappa, Specificity, Sensitivity, AUC-ROC, Log-Loss, etc.)
   3. Graphs & Curves (Confusion Matrix, ROC, Precision-Recall,
      F1-vs-Threshold, Training/Loss-vs-Epochs, Per-class bar charts)
   4. Grad-CAM Heatmaps for prediction visualization
============================================================================
"""

# ========================= 0. IMPORTS ========================================
import os
import sys
import random
import warnings
import json
from pathlib import Path
from collections import OrderedDict

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# We need PennyLane for the quantum layers
try:
    import pennylane as qml
    PL_AVAILABLE = True
except ImportError:
    print("[ERROR] PennyLane is required. Install with: pip install pennylane pennylane-lightning")
    sys.exit(1)

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, log_loss,
    matthews_corrcoef, cohen_kappa_score, roc_auc_score,
    multilabel_confusion_matrix,
)

warnings.filterwarnings("ignore")

# ========================= 1. CONFIGURATION ==================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

PROJECT_ROOT = Path(r"E:\Project_with_hybrid\appleleaf.v2i.yolo26-project")
DATASET_ROOT = PROJECT_ROOT / "dataset"
YAML_PATH    = DATASET_ROOT / "data.yaml"
MODEL_PATH   = PROJECT_ROOT / "models" / "QResnet" / "hybrid_quantum_resnet_best.pth"
METRICS_JSON = PROJECT_ROOT / "models" / "QResnet" / "qresnet_metrics.json"
OUTPUT_DIR   = PROJECT_ROOT / "result_Qresnet" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {DEVICE}")

# Disease class names (mapped from numeric IDs in data.yaml)
DISEASE_NAMES = [
    "Alternaria Leaf Spot",
    "Brown Spot",
    "Frogeye Leaf Spot",
    "Grey Spot",
    "Healthy",
    "Mosaic",
    "Powdery Mildew",
    "Rust",
    "Scab",
]

# ========================= 2. DATASET ========================================
with open(YAML_PATH, "r") as f:
    data_cfg = yaml.safe_load(f)

NUM_CLASSES = data_cfg["nc"]  # 9


class YOLOClassificationDataset(Dataset):
    """Loads images from YOLO-format directory with labels from .txt files."""

    def __init__(self, split, transform=None):
        key_map = {"train": "train", "test": "test", "val": "val"}
        rel = data_cfg.get(key_map.get(split, split), "")
        img_dir = (YAML_PATH.parent / rel.replace("./", "")).resolve()
        if not img_dir.exists():
            raise FileNotFoundError(f"Image dir not found: {img_dir}")

        self.img_files = sorted(
            list(img_dir.glob("*.jpg"))
            + list(img_dir.glob("*.png"))
            + list(img_dir.glob("*.jpeg"))
        )
        self.label_dir = img_dir.parent / "labels"
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        image = Image.open(img_path).convert("RGB")

        label_path = self.label_dir / (img_path.stem + ".txt")
        class_id = 0
        if label_path.exists():
            with open(label_path, "r") as f:
                tok = f.readline().strip().split()
                if tok:
                    class_id = int(tok[0])

        if self.transform:
            image = self.transform(image)

        return image, class_id, str(img_path)


# ImageNet normalization (same as used during training)
imagenet_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ========================= 3. MODEL DEFINITION ===============================
# Exact same architecture as in evaluate_all_models.py

N_QUBITS_QR = 8
_dev_qr = qml.device("default.qubit", wires=N_QUBITS_QR)


@qml.qnode(_dev_qr, interface="torch")
def _qnode_qr(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(N_QUBITS_QR))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(N_QUBITS_QR))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS_QR)]


_ws_qr = {"weights": (4, N_QUBITS_QR, 3)}
_ql_qr = qml.qnn.TorchLayer(_qnode_qr, _ws_qr)


class QuantumFiLM(nn.Module):
    """Feature-wise Linear Modulation with quantum processing."""
    def __init__(self):
        super().__init__()
        self.in_channels = 512
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(512, N_QUBITS_QR), nn.Tanh(),
        )
        self.fc = nn.Sequential(
            nn.Linear(N_QUBITS_QR, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 512 * 2),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        pooled = self.attention(x) * np.pi
        q_out = _ql_qr(pooled).float()
        film_params = self.fc(q_out)
        scale, shift = torch.chunk(film_params, 2, dim=1)
        scale = scale.view(B, C, 1, 1)
        shift = shift.view(B, C, 1, 1)
        return x * (1 + scale.tanh()) + shift.tanh()


class HybridQuantumResNet(nn.Module):
    """ResNet50 backbone with Quantum FiLM modulation.

    Architecture matches the saved checkpoint which keeps `self.backbone`
    (full ResNet50) AND builds `layer1`/`layer2` Sequentials from its
    sub-modules, so both `backbone.*` and `layer1.*`/`layer2.*` state_dict
    keys are present.
    """
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        self.layer1 = nn.Sequential(
            self.backbone.conv1, self.backbone.bn1, self.backbone.relu,
            self.backbone.maxpool, self.backbone.layer1, self.backbone.layer2,
        )
        self.quantum_film = QuantumFiLM()
        self.layer2 = nn.Sequential(self.backbone.layer3, self.backbone.layer4)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 9),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.quantum_film(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        return self.head(x)


# ========================= 4. LOAD MODEL =====================================
print("\n" + "=" * 70)
print("  LOADING HYBRID QUANTUM RESNET MODEL")
print("=" * 70)

model = HybridQuantumResNet()
checkpoint = torch.load(str(MODEL_PATH), map_location="cpu")

# Load model weights
model.load_state_dict(checkpoint["model_state"])
model.eval().to(DEVICE)

# Extract training history if available
train_history = {}
for key in ["train_losses", "val_losses", "train_accs", "val_accs",
            "train_loss", "val_loss", "train_acc", "val_acc",
            "history", "epoch", "epochs"]:
    if key in checkpoint:
        train_history[key] = checkpoint[key]

print(f"[INFO] Model loaded successfully from: {MODEL_PATH}")
print(f"[INFO] Checkpoint keys: {list(checkpoint.keys())}")
print(f"[INFO] Training history keys found: {list(train_history.keys())}")


# ========================= 5. MODEL ARCHITECTURE ============================
print("\n" + "=" * 70)
print("  MODEL ARCHITECTURE")
print("=" * 70)

# Full architecture printout
arch_str = str(model)
print(arch_str)

# Save architecture to file
with open(OUTPUT_DIR / "model_architecture.txt", "w", encoding="utf-8") as f:
    f.write("=" * 70 + "\n")
    f.write("  HYBRID QUANTUM RESNET - FULL ARCHITECTURE\n")
    f.write("=" * 70 + "\n\n")
    f.write(arch_str)
    f.write("\n\n")

    # Parameter count
    f.write("=" * 70 + "\n")
    f.write("  PARAMETER SUMMARY\n")
    f.write("=" * 70 + "\n\n")

    total_params = 0
    trainable_params = 0
    layer_info = []
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        layer_info.append((name, list(param.shape), param.numel(), param.requires_grad))

    for name, shape, count, trainable in layer_info:
        f.write(f"  {name:60s} | Shape: {str(shape):30s} | Params: {count:>10,d} | Trainable: {trainable}\n")

    f.write(f"\n{'-' * 70}\n")
    f.write(f"  Total Parameters:      {total_params:>15,d}\n")
    f.write(f"  Trainable Parameters:  {trainable_params:>15,d}\n")
    f.write(f"  Non-trainable Params:  {total_params - trainable_params:>15,d}\n")
    f.write(f"  Model Size (MB):       {total_params * 4 / (1024**2):>15.2f}\n")

print(f"\n  Total Parameters:      {total_params:>12,d}")
print(f"  Trainable Parameters:  {trainable_params:>12,d}")
print(f"  Model Size (approx):   {total_params * 4 / (1024**2):.2f} MB")
print(f"\n[INFO] Architecture saved to: {OUTPUT_DIR / 'model_architecture.txt'}")


# ========================= 6. INFERENCE ======================================
print("\n" + "=" * 70)
print("  RUNNING INFERENCE ON ALL SPLITS")
print("=" * 70)


def run_inference(model, split_name, transform, batch_size=8):
    """Run model inference on a dataset split. Returns true/pred labels + probabilities."""
    dataset = YOLOClassificationDataset(split_name, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    y_true, y_pred, y_probs, all_paths = [], [], [], []
    import time
    total_time = 0.0

    with torch.no_grad():
        for imgs, labels, paths in tqdm(loader, desc=f"  [{split_name.upper()}]", leave=True):
            imgs = imgs.to(DEVICE)
            t0 = time.perf_counter()
            logits = model(imgs)
            total_time += time.perf_counter() - t0

            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

            y_true.extend(labels.numpy())
            y_pred.extend(preds)
            y_probs.extend(probs)
            all_paths.extend(paths)

    avg_ms = (total_time / len(dataset)) * 1000 if len(dataset) > 0 else 0
    return (np.array(y_true), np.array(y_pred),
            np.array(y_probs), all_paths, avg_ms)


# Run inference on all splits
results = {}
for split in ["train", "test", "val"]:
    try:
        yt, yp, yprobs, paths, inf_ms = run_inference(model, split, imagenet_tf, batch_size=8)
        results[split] = {
            "y_true": yt, "y_pred": yp, "y_probs": yprobs,
            "paths": paths, "inf_time_ms": inf_ms,
        }
        print(f"  [OK] {split.upper()} - {len(yt)} samples | "
              f"Acc: {accuracy_score(yt, yp):.4f} | Inf: {inf_ms:.2f} ms/img")
    except Exception as e:
        print(f"  [FAIL] {split.upper()} - Error: {e}")


# ========================= 7. COMPREHENSIVE METRICS ==========================
print("\n" + "=" * 70)
print("  COMPREHENSIVE PERFORMANCE METRICS")
print("=" * 70)


def compute_all_metrics(y_true, y_pred, y_probs, split_name):
    """Compute every possible classification metric."""
    metrics = OrderedDict()

    # Overall metrics
    metrics["Accuracy"] = accuracy_score(y_true, y_pred)
    metrics["Precision (Macro)"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["Precision (Weighted)"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["Precision (Micro)"] = precision_score(y_true, y_pred, average="micro", zero_division=0)
    metrics["Recall (Macro)"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["Recall (Weighted)"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["Recall (Micro)"] = recall_score(y_true, y_pred, average="micro", zero_division=0)
    metrics["F1-Score (Macro)"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["F1-Score (Weighted)"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["F1-Score (Micro)"] = f1_score(y_true, y_pred, average="micro", zero_division=0)
    metrics["F2-Score (Macro)"] = fbeta_score(y_true, y_pred, beta=2, average="macro", zero_division=0)
    metrics["F2-Score (Weighted)"] = fbeta_score(y_true, y_pred, beta=2, average="weighted", zero_division=0)
    metrics["F0.5-Score (Macro)"] = fbeta_score(y_true, y_pred, beta=0.5, average="macro", zero_division=0)
    metrics["Matthews Corr. Coeff. (MCC)"] = matthews_corrcoef(y_true, y_pred)
    metrics["Cohen's Kappa"] = cohen_kappa_score(y_true, y_pred)

    # Log loss
    try:
        metrics["Log Loss"] = log_loss(y_true, y_probs, labels=list(range(NUM_CLASSES)))
    except Exception:
        metrics["Log Loss"] = float("nan")

    # AUC-ROC (One-vs-Rest)
    try:
        y_true_onehot = np.eye(NUM_CLASSES)[y_true]
        metrics["AUC-ROC (Macro OvR)"] = roc_auc_score(
            y_true_onehot, y_probs, average="macro", multi_class="ovr"
        )
        metrics["AUC-ROC (Weighted OvR)"] = roc_auc_score(
            y_true_onehot, y_probs, average="weighted", multi_class="ovr"
        )
    except Exception:
        metrics["AUC-ROC (Macro OvR)"] = float("nan")
        metrics["AUC-ROC (Weighted OvR)"] = float("nan")

    # Per-class specificity and sensitivity from multilabel confusion matrix
    mcm = multilabel_confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    specificities = []
    sensitivities = []
    for i in range(NUM_CLASSES):
        tn, fp, fn, tp = mcm[i].ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificities.append(spec)
        sensitivities.append(sens)

    metrics["Specificity (Macro)"] = np.mean(specificities)
    metrics["Sensitivity (Macro)"] = np.mean(sensitivities)  # same as Recall Macro

    # Top-K accuracy
    for k in [1, 3, 5]:
        if k <= NUM_CLASSES:
            top_k = np.array([
                y_true[i] in np.argsort(y_probs[i])[-k:]
                for i in range(len(y_true))
            ]).mean()
            metrics[f"Top-{k} Accuracy"] = top_k

    return metrics


# Compute and display metrics for each split
all_metrics = {}
for split_name, data in results.items():
    metrics = compute_all_metrics(
        data["y_true"], data["y_pred"], data["y_probs"], split_name
    )
    metrics["Inference Time (ms/img)"] = data["inf_time_ms"]
    metrics["Total Samples"] = len(data["y_true"])
    all_metrics[split_name] = metrics

    print(f"\n{'-' * 60}")
    print(f"  {split_name.upper()} SET METRICS")
    print(f"{'-' * 60}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:40s} : {v:.6f}")
        else:
            print(f"  {k:40s} : {v}")

# Per-class classification report (for test set)
if "test" in results:
    print(f"\n{'-' * 60}")
    print(f"  PER-CLASS CLASSIFICATION REPORT (TEST SET)")
    print(f"{'-' * 60}")
    report = classification_report(
        results["test"]["y_true"],
        results["test"]["y_pred"],
        target_names=DISEASE_NAMES,
        digits=4,
        zero_division=0,
    )
    print(report)

# Save all metrics to JSON
metrics_save = {}
for split, m in all_metrics.items():
    metrics_save[split] = {k: float(v) if isinstance(v, (float, np.floating)) else int(v) for k, v in m.items()}

with open(OUTPUT_DIR / "qresnet_full_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics_save, f, indent=4)

print(f"\n[INFO] Full metrics saved to: {OUTPUT_DIR / 'qresnet_full_metrics.json'}")


## ========================= 8. PLOTS & GRAPHS ================================
print("\n" + "=" * 70)
print("  GENERATING UPGRADED PLOTS & GRAPHS")
print("=" * 70)

# Style configuration
plt.rcParams.update({
    "font.size": 11,
    "figure.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
    "figure.facecolor": "white",
})
sns.set_style("whitegrid")

# Global colors, markers, and line styles for class-wise distinctiveness
COLORS = [
    "#0E8388",  # Deep Teal
    "#FF6E61",  # Electric Coral
    "#7048E8",  # Royal Purple
    "#20C997",  # Vibrant Green
    "#F59F00",  # Warm Gold
    "#1098AD",  # Bright Cyan
    "#E64980",  # Rose Pink
    "#FD7E14",  # Bright Orange
    "#3B5BDB"   # Indigo Blue
]
MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p", "*"]
LINE_STYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-"]


# ---- 8a. Confusion Matrices ------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, split_name, normalize=False):
    """Plot and save an upgraded confusion matrix with counts and percentages."""
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    
    # Compute the percentages relative to the true class support (row sums)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)  # avoid division by zero
    percentages = cm.astype("float") / row_sums
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Display both Count and Percentage in each cell
    annot_labels = np.empty_like(cm, dtype=object)
    for r in range(NUM_CLASSES):
        for c in range(NUM_CLASSES):
            cnt = cm[r, c]
            pct = percentages[r, c] * 100
            if cnt > 0:
                annot_labels[r, c] = f"{cnt}\n({pct:.1f}%)"
            else:
                annot_labels[r, c] = "0\n(0.0%)"
                
    color_matrix = percentages if normalize else cm
    cmap_name = "crest" if normalize else "mako"
    
    sns.heatmap(
        color_matrix, annot=annot_labels, fmt="", cmap=cmap_name,
        xticklabels=DISEASE_NAMES, yticklabels=DISEASE_NAMES,
        ax=ax, linewidths=1.5, linecolor="#FAF9F6",
        cbar_kws={"shrink": 0.8, "label": "Proportion (Normalized)" if normalize else "Sample Count"},
        annot_kws={"size": 10, "fontweight": "bold"},
    )
    
    title_suffix = "(Normalized by Class Support)" if normalize else "(Absolute Counts)"
    ax.set_title(f"Confusion Matrix — {split_name.upper()} {title_suffix}",
                 fontsize=15, fontweight="bold", pad=20, color="#2D3748")
    ax.set_xlabel("Predicted Disease Class", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("True Disease Class", fontsize=12, fontweight="bold", labelpad=10)
    ax.tick_params(axis="x", rotation=40, labelsize=10)
    ax.tick_params(axis="y", rotation=0, labelsize=10)
    
    # Add border outlines
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color("#CCCCCC")
        
    plt.tight_layout()

    norm_str = "_normalized" if normalize else ""
    fname = f"confusion_matrix_{split_name}{norm_str}.png"
    fig.savefig(OUTPUT_DIR / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved: {fname}")


for split_name, data in results.items():
    plot_confusion_matrix(data["y_true"], data["y_pred"], split_name, normalize=False)
    plot_confusion_matrix(data["y_true"], data["y_pred"], split_name, normalize=True)


# ---- 8b. ROC Curves (One-vs-Rest) -----------------------------------------
def plot_roc_curves(y_true, y_probs, split_name):
    """Plot ROC curves for all classes + macro average with an inset zoom plot."""
    y_true_onehot = np.eye(NUM_CLASSES)[y_true]

    fig, ax = plt.subplots(figsize=(11, 9))
    
    # We will collect fpr/tpr to compute macro averages
    all_fpr = np.linspace(0, 1, 200)
    mean_tpr = np.zeros_like(all_fpr)

    # Inset axes to zoom in on top-left (high-performance) region
    ax_ins = ax.inset_axes([0.42, 0.15, 0.48, 0.45])
    
    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_probs[:, i])
        roc_auc_val = auc(fpr, tpr)
        
        # Main plot
        ax.plot(fpr, tpr, color=COLORS[i], lw=2, linestyle=LINE_STYLES[i],
                marker=MARKERS[i], markevery=max(1, len(fpr)//20), markersize=5, alpha=0.85,
                label=f"{DISEASE_NAMES[i]} (AUC = {roc_auc_val:.4f})")
        
        # Inset plot
        ax_ins.plot(fpr, tpr, color=COLORS[i], lw=1.5, linestyle=LINE_STYLES[i],
                    marker=MARKERS[i], markevery=max(1, len(fpr)//20), markersize=4, alpha=0.85)
        
        mean_tpr += np.interp(all_fpr, fpr, tpr)

    mean_tpr /= NUM_CLASSES
    mean_auc = auc(all_fpr, mean_tpr)
    
    # Plot Macro Average
    ax.plot(all_fpr, mean_tpr, color="#1A1A1A", linestyle="--", lw=3.0, alpha=0.95,
            label=f"Macro Average (AUC = {mean_auc:.4f})")
    ax_ins.plot(all_fpr, mean_tpr, color="#1A1A1A", linestyle="--", lw=2.0, alpha=0.95)
    
    # Baselines
    ax.plot([0, 1], [0, 1], ":", color="grey", lw=1.5, alpha=0.5)
    ax_ins.plot([0, 1], [0, 1], ":", color="grey", lw=1.5, alpha=0.5)

    # Find Youden's J-statistic on macro curve to show the optimal operating threshold
    optimal_idx = np.argmax(mean_tpr - all_fpr)
    opt_fpr = all_fpr[optimal_idx]
    opt_tpr = mean_tpr[optimal_idx]
    
    # Annotate optimal operating threshold
    ax.plot(opt_fpr, opt_tpr, marker="*", color="#E63946", markersize=14, markeredgecolor="black", markeredgewidth=1.5, zorder=10)
    ax.annotate(f"Optimal Threshold\nFPR={opt_fpr:.2f}, TPR={opt_tpr:.2f}",
                xy=(opt_fpr, opt_tpr), xytext=(opt_fpr + 0.08, opt_tpr - 0.15),
                arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=6, headlength=6),
                fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="#FAF9F6", edgecolor="grey", alpha=0.9))

    # Inset zoom styling
    ax_ins.set_xlim(-0.01, 0.20)
    ax_ins.set_ylim(0.80, 1.01)
    ax_ins.grid(True, linestyle=":", alpha=0.6)
    ax.indicate_inset_zoom(ax_ins, edgecolor="#4A5568")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_title(f"ROC Curves (One-vs-Rest) — {split_name.upper()}", fontsize=15, fontweight="bold", pad=15, color="#2D3748")
    
    # Legend placement and formatting
    ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.45), fontsize=8.5, framealpha=0.95, facecolor="#FAF9F6", edgecolor="#E2E8F0")
    ax.grid(True, linestyle="--", alpha=0.5)
    
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color("#CCCCCC")
        
    plt.tight_layout()

    fname = f"roc_curves_{split_name}.png"
    fig.savefig(OUTPUT_DIR / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved: {fname}")


for split_name, data in results.items():
    plot_roc_curves(data["y_true"], data["y_probs"], split_name)


# ---- 8c. Precision-Recall Curves -------------------------------------------
def plot_pr_curves(y_true, y_probs, split_name):
    """Plot Precision-Recall curves with F1-contours and an inset zoom plot."""
    y_true_onehot = np.eye(NUM_CLASSES)[y_true]

    fig, ax = plt.subplots(figsize=(11, 9))
    
    # Inset axes to zoom in on top-right (high-performance) region
    ax_ins = ax.inset_axes([0.15, 0.15, 0.45, 0.45])

    # Plot iso-F1 contours in the background to show performance boundaries
    f_scores = np.linspace(0.2, 0.8, 4)
    for f_score in f_scores:
        x_vals = np.linspace(0.01, 1.0, 100)
        y_vals = f_score * x_vals / (2 * x_vals - f_score)
        valid = (y_vals >= 0) & (y_vals <= 1)
        ax.plot(x_vals[valid], y_vals[valid], color="gray", alpha=0.2, lw=1, linestyle=":")
        ax.annotate(f"f1={f_score:.1f}", xy=(0.9, y_vals[valid][-10] if sum(valid) > 10 else 0.9), color="gray", alpha=0.4, fontsize=8)

    for i in range(NUM_CLASSES):
        precision_vals, recall_vals, _ = precision_recall_curve(
            y_true_onehot[:, i], y_probs[:, i]
        )
        ap = average_precision_score(y_true_onehot[:, i], y_probs[:, i])
        
        # Main plot
        ax.plot(recall_vals, precision_vals, color=COLORS[i], lw=2, linestyle=LINE_STYLES[i],
                marker=MARKERS[i], markevery=max(1, len(recall_vals)//20), markersize=5, alpha=0.85,
                label=f"{DISEASE_NAMES[i]} (AP = {ap:.4f})")
        
        # Inset plot
        ax_ins.plot(recall_vals, precision_vals, color=COLORS[i], lw=1.5, linestyle=LINE_STYLES[i],
                    marker=MARKERS[i], markevery=max(1, len(recall_vals)//20), markersize=4, alpha=0.85)

    # Inset zoom styling
    ax_ins.set_xlim(0.78, 1.02)
    ax_ins.set_ylim(0.78, 1.02)
    ax_ins.grid(True, linestyle=":", alpha=0.6)
    ax.indicate_inset_zoom(ax_ins, edgecolor="#4A5568")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("Recall (Sensitivity)", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_ylabel("Precision (PPV)", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_title(f"Precision-Recall Curves — {split_name.upper()}", fontsize=15, fontweight="bold", pad=15, color="#2D3748")
    
    # Legend placement
    ax.legend(loc="upper left", bbox_to_anchor=(0.55, 0.95), fontsize=8.5, framealpha=0.95, facecolor="#FAF9F6", edgecolor="#E2E8F0")
    ax.grid(True, linestyle="--", alpha=0.5)
    
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color("#CCCCCC")
        
    plt.tight_layout()

    fname = f"precision_recall_curves_{split_name}.png"
    fig.savefig(OUTPUT_DIR / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved: {fname}")


for split_name, data in results.items():
    plot_pr_curves(data["y_true"], data["y_probs"], split_name)


# ---- 8d. Per-class Metrics Bar Charts --------------------------------------
def plot_per_class_metrics(y_true, y_pred, split_name):
    """Plot precision, recall, F1 per class using horizontal grouped bar charts."""
    prec_per = precision_score(y_true, y_pred, average=None, zero_division=0, labels=range(NUM_CLASSES))
    rec_per = recall_score(y_true, y_pred, average=None, zero_division=0, labels=range(NUM_CLASSES))
    f1_per = f1_score(y_true, y_pred, average=None, zero_division=0, labels=range(NUM_CLASSES))

    y = np.arange(NUM_CLASSES)
    height = 0.25

    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Draw bars horizontally for better class label readability
    bars1 = ax.barh(y + height, prec_per, height, label="Precision", color="#1098AD", alpha=0.9)
    bars2 = ax.barh(y, rec_per, height, label="Recall", color="#20C997", alpha=0.9)
    bars3 = ax.barh(y - height, f1_per, height, label="F1-Score", color="#7048E8", alpha=0.9)

    # Value labels on top of bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            w = bar.get_width()
            if w > 0:
                ax.text(w + 0.01, bar.get_y() + bar.get_height() / 2.,
                        f"{w:.3f}", ha="left", va="center", fontsize=8, fontweight="bold", color="#2D3748")

    # Add reference lines for macro-average metrics
    macro_prec = np.mean(prec_per)
    macro_rec = np.mean(rec_per)
    macro_f1 = np.mean(f1_per)
    
    ax.axvline(macro_prec, color="#1098AD", linestyle="--", alpha=0.6, lw=1.2, label=f"Avg Precision ({macro_prec:.3f})")
    ax.axvline(macro_rec, color="#20C997", linestyle="--", alpha=0.6, lw=1.2, label=f"Avg Recall ({macro_rec:.3f})")
    ax.axvline(macro_f1, color="#7048E8", linestyle="--", alpha=0.6, lw=1.2, label=f"Avg F1-Score ({macro_f1:.3f})")

    ax.set_yticks(y)
    ax.set_yticklabels(DISEASE_NAMES, fontsize=10, fontweight="bold")
    ax.set_xlabel("Score", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_title(f"Per-Class Performance Metrics — {split_name.upper()}", fontsize=15, fontweight="bold", pad=15, color="#2D3748")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95, facecolor="#FAF9F6", edgecolor="#E2E8F0")
    ax.set_xlim(0, 1.15)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color("#CCCCCC")
        
    plt.tight_layout()

    fname = f"per_class_metrics_{split_name}.png"
    fig.savefig(OUTPUT_DIR / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved: {fname}")


for split_name, data in results.items():
    plot_per_class_metrics(data["y_true"], data["y_pred"], split_name)


# ---- 8e. Train vs Test Accuracy & F1 Comparison ----------------------------
def plot_train_test_comparison():
    """Side-by-side comparison of train vs test metrics using a horizontal gap plot."""
    if "train" not in results or "test" not in results:
        return

    train_data = results["train"]
    test_data = results["test"]

    metrics_names = ["Accuracy", "Precision (Weighted)", "Recall (Weighted)",
                     "F1 (Weighted)", "F2 (Weighted)", "MCC", "Cohen's Kappa"]
    train_vals = [
        accuracy_score(train_data["y_true"], train_data["y_pred"]),
        precision_score(train_data["y_true"], train_data["y_pred"], average="weighted", zero_division=0),
        recall_score(train_data["y_true"], train_data["y_pred"], average="weighted", zero_division=0),
        f1_score(train_data["y_true"], train_data["y_pred"], average="weighted", zero_division=0),
        fbeta_score(train_data["y_true"], train_data["y_pred"], beta=2, average="weighted", zero_division=0),
        matthews_corrcoef(train_data["y_true"], train_data["y_pred"]),
        cohen_kappa_score(train_data["y_true"], train_data["y_pred"]),
    ]
    test_vals = [
        accuracy_score(test_data["y_true"], test_data["y_pred"]),
        precision_score(test_data["y_true"], test_data["y_pred"], average="weighted", zero_division=0),
        recall_score(test_data["y_true"], test_data["y_pred"], average="weighted", zero_division=0),
        f1_score(test_data["y_true"], test_data["y_pred"], average="weighted", zero_division=0),
        fbeta_score(test_data["y_true"], test_data["y_pred"], beta=2, average="weighted", zero_division=0),
        matthews_corrcoef(test_data["y_true"], test_data["y_pred"]),
        cohen_kappa_score(test_data["y_true"], test_data["y_pred"]),
    ]

    y = np.arange(len(metrics_names))
    height = 0.3

    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot horizontal bars
    bars1 = ax.barh(y + height/2, train_vals, height, label="Train Set", color="#3B5BDB", alpha=0.9)
    bars2 = ax.barh(y - height/2, test_vals, height, label="Test Set", color="#FD7E14", alpha=0.9)

    # Annotate values and generalization gaps
    for idx, (t_val, s_val) in enumerate(zip(train_vals, test_vals)):
        ax.text(t_val + 0.01, idx + height/2, f"{t_val:.4f}", ha="left", va="center", fontsize=8, fontweight="bold", color="#3B5BDB")
        ax.text(s_val + 0.01, idx - height/2, f"{s_val:.4f}", ha="left", va="center", fontsize=8, fontweight="bold", color="#FD7E14")
        
        gap = t_val - s_val
        gap_color = "#E64980" if gap > 0.05 else "#20C997"
        # Draw connecting gap bar
        ax.plot([s_val, t_val], [idx, idx], color=gap_color, lw=2.5, marker="|", markersize=8)
        ax.text((s_val + t_val)/2, idx + 0.12, f"Gap: {gap:+.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold", color=gap_color)

    ax.set_yticks(y)
    ax.set_yticklabels(metrics_names, fontsize=10, fontweight="bold")
    ax.set_xlabel("Metric Value", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_title("Generalization Capacity — Train vs Test Metrics Comparison", fontsize=15, fontweight="bold", pad=15, color="#2D3748")
    ax.legend(loc="lower left", fontsize=10, framealpha=0.95, facecolor="#FAF9F6", edgecolor="#E2E8F0")
    ax.set_xlim(0, 1.2)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color("#CCCCCC")
        
    plt.tight_layout()

    fig.savefig(OUTPUT_DIR / "train_vs_test_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] Saved: train_vs_test_comparison.png")


plot_train_test_comparison()


# ---- 8f. Training History (Loss & Accuracy vs Epochs) ----------------------
def plot_training_history():
    """Plot training history with smooth gradient fills and best-epoch indicators."""
    has_history = False
    loss_keys = [k for k in train_history if "loss" in k.lower()]
    acc_keys = [k for k in train_history if "acc" in k.lower()]

    if loss_keys or acc_keys:
        has_history = True
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        val_loss_key = None
        for key in loss_keys:
            if "val" in key.lower():
                val_loss_key = key
                break
        
        best_epoch = None
        if val_loss_key and isinstance(train_history[val_loss_key], (list, np.ndarray)):
            best_epoch = np.argmin(train_history[val_loss_key]) + 1

        # Loss plot
        ax1 = axes[0]
        for key in loss_keys:
            data_vals = train_history[key]
            if isinstance(data_vals, (list, np.ndarray)):
                epochs = np.arange(1, len(data_vals) + 1)
                label = key.replace("_", " ").title()
                line_color = "#E64980" if "val" in key.lower() else "#3B5BDB"
                ax1.plot(epochs, data_vals, linewidth=2.5, label=label, marker="o", markersize=4, color=line_color)
                ax1.fill_between(epochs, data_vals, alpha=0.08, color=line_color)

        if best_epoch:
            ax1.axvline(best_epoch, color="#F59F00", linestyle="--", lw=1.5, label=f"Best Model (Epoch {best_epoch})")
            best_val_loss = train_history[val_loss_key][best_epoch - 1]
            ax1.plot(best_epoch, best_val_loss, marker="*", color="#F59F00", markersize=12, markeredgecolor="black")
            ax1.text(best_epoch + 0.5, best_val_loss + 0.05, f"Min Loss: {best_val_loss:.4f}", fontsize=9, fontweight="bold", color="#F59F00")

        ax1.set_xlabel("Training Epoch", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Loss Value", fontsize=12, fontweight="bold")
        ax1.set_title("Cross-Entropy Loss vs Epochs", fontsize=14, fontweight="bold", color="#2D3748")
        ax1.legend(fontsize=10, framealpha=0.95, facecolor="#FAF9F6", edgecolor="#E2E8F0")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Accuracy plot
        ax2 = axes[1]
        val_acc_key = None
        for key in acc_keys:
            if "val" in key.lower():
                val_acc_key = key
            data_vals = train_history[key]
            if isinstance(data_vals, (list, np.ndarray)):
                epochs = np.arange(1, len(data_vals) + 1)
                label = key.replace("_", " ").title()
                line_color = "#20C997" if "val" in key.lower() else "#7048E8"
                ax2.plot(epochs, data_vals, linewidth=2.5, label=label, marker="o", markersize=4, color=line_color)
                ax2.fill_between(epochs, data_vals, alpha=0.08, color=line_color)

        if best_epoch and val_acc_key and isinstance(train_history[val_acc_key], (list, np.ndarray)):
            ax2.axvline(best_epoch, color="#F59F00", linestyle="--", lw=1.5, label=f"Best Model (Epoch {best_epoch})")
            best_val_acc = train_history[val_acc_key][best_epoch - 1]
            ax2.plot(best_epoch, best_val_acc, marker="*", color="#F59F00", markersize=12, markeredgecolor="black")
            ax2.text(best_epoch + 0.5, best_val_acc - 2.0, f"Max Acc: {best_val_acc:.2f}%", fontsize=9, fontweight="bold", color="#F59F00")

        ax2.set_xlabel("Training Epoch", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
        ax2.set_title("Classification Accuracy vs Epochs", fontsize=14, fontweight="bold", color="#2D3748")
        ax2.legend(fontsize=10, framealpha=0.95, facecolor="#FAF9F6", edgecolor="#E2E8F0")
        ax2.grid(True, linestyle="--", alpha=0.5)

        plt.suptitle("QResNet — Training History Optimization", fontsize=16, fontweight="bold", y=0.98, color="#2D3748")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(OUTPUT_DIR / "training_history.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("  [OK] Saved: training_history.png")

    if not has_history and METRICS_JSON.exists():
        print("  [INFO] No epoch-level history in checkpoint. Using saved metrics JSON for summary.")
        with open(METRICS_JSON, "r") as f:
            saved_metrics = json.load(f)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        if "train" in saved_metrics and "test" in saved_metrics:
            splits = ["Train Set", "Test Set"]
            losses = [saved_metrics["train"].get("loss", 0), saved_metrics["test"].get("loss", 0)]
            accs = [saved_metrics["train"].get("accuracy", 0), saved_metrics["test"].get("accuracy", 0)]

            ax1 = axes[0]
            bars1 = ax1.bar(splits, losses, color=["#3B5BDB", "#FD7E14"], alpha=0.85, width=0.45)
            for bar, val in zip(bars1, losses):
                ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                         f"{val:.4f}", ha="center", fontsize=11, fontweight="bold", color="#2D3748")
            ax1.set_ylabel("Cross-Entropy Loss", fontsize=12, fontweight="bold")
            ax1.set_title("Train vs Test Loss (Final)", fontsize=14, fontweight="bold", color="#2D3748")
            ax1.grid(axis="y", linestyle="--", alpha=0.5)

            ax2 = axes[1]
            bars2 = ax2.bar(splits, accs, color=["#3B5BDB", "#FD7E14"], alpha=0.85, width=0.45)
            for bar, val in zip(bars2, accs):
                ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                         f"{val:.2f}%", ha="center", fontsize=11, fontweight="bold", color="#2D3748")
            ax2.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
            ax2.set_title("Train vs Test Accuracy (Final)", fontsize=14, fontweight="bold", color="#2D3748")
            ax2.grid(axis="y", linestyle="--", alpha=0.5)

            plt.suptitle("QResNet — Final Performance Comparison Summary", fontsize=16, fontweight="bold", y=0.98, color="#2D3748")
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            fig.savefig(OUTPUT_DIR / "training_summary.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
            print("  [OK] Saved: training_summary.png")


plot_training_history()


# ---- 8g. Per-class Accuracy Heatmap ----------------------------------------
def plot_per_class_accuracy_heatmap():
    """Heatmap of per-class accuracy across splits with custom colorbar styling."""
    split_names = []
    pca_matrix = []

    for split_name, data in results.items():
        cm = confusion_matrix(data["y_true"], data["y_pred"], labels=range(NUM_CLASSES))
        row_sums = cm.sum(axis=1)
        per_class_acc = np.where(row_sums > 0, cm.diagonal() / row_sums, 0)
        pca_matrix.append(per_class_acc)
        split_names.append(split_name.upper())

    pca_matrix = np.array(pca_matrix)

    fig, ax = plt.subplots(figsize=(13, 5))
    sns.heatmap(
        pca_matrix, annot=True, fmt=".3f", cmap="mako_r",
        xticklabels=DISEASE_NAMES, yticklabels=split_names,
        ax=ax, vmin=0, vmax=1, linewidths=1.5, linecolor="#FAF9F6",
        annot_kws={"size": 11, "fontweight": "bold"},
        cbar_kws={"shrink": 0.8, "label": "Per-class Accuracy Rate"},
    )
    ax.set_title("Per-Class Accuracy Heatmap Across Data Splits", fontsize=15, fontweight="bold", pad=15, color="#2D3748")
    ax.set_xlabel("Disease Class", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_ylabel("Dataset Split", fontsize=12, fontweight="bold", labelpad=8)
    ax.tick_params(axis="x", rotation=40, labelsize=10)
    ax.tick_params(axis="y", rotation=0, labelsize=10)
    
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color("#CCCCCC")
        
    plt.tight_layout()

    fig.savefig(OUTPUT_DIR / "per_class_accuracy_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] Saved: per_class_accuracy_heatmap.png")


plot_per_class_accuracy_heatmap()


# ---- 8h. Class Distribution ------------------------------------------------
def plot_class_distribution():
    """Plot consolidated class distribution showing Train, Val, Test side-by-side."""
    y = np.arange(NUM_CLASSES)
    height = 0.25

    fig, ax = plt.subplots(figsize=(13, 9))
    
    # Draw horizontal bars for splits side-by-side to directly compare sample sizes
    split_colors = {"train": "#3B5BDB", "val": "#20C997", "test": "#FD7E14"}
    
    for idx, split_name in enumerate(["train", "val", "test"]):
        if split_name not in results:
            continue
        data = results[split_name]
        unique, counts = np.unique(data["y_true"], return_counts=True)
        class_counts = np.zeros(NUM_CLASSES, dtype=int)
        for u, c in zip(unique, counts):
            class_counts[u] = c
            
        offset = (idx - 1) * height
        bars = ax.barh(y + offset, class_counts, height, label=f"{split_name.upper()} (Total: {sum(class_counts)})",
                        color=split_colors[split_name], alpha=0.9)
        
        # Add text label to the right of each bar
        for bar in bars:
            w = bar.get_width()
            if w > 0:
                ax.text(w + 0.5, bar.get_y() + bar.get_height()/2, str(int(w)),
                        ha="left", va="center", fontsize=8, fontweight="bold", color="#2D3748")

    ax.set_yticks(y)
    ax.set_yticklabels(DISEASE_NAMES, fontsize=10, fontweight="bold")
    ax.set_xlabel("Number of Samples", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_title("Consolidated Dataset Class Distribution Across Splits", fontsize=15, fontweight="bold", pad=15, color="#2D3748")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95, facecolor="#FAF9F6", edgecolor="#E2E8F0")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color("#CCCCCC")
        
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "class_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] Saved: class_distribution.png")


plot_class_distribution()


# ---- 8i. Confidence Distribution Histogram ----------------------------------
def plot_confidence_distribution():
    """Histogram of prediction confidence (max probability) split by correct/incorrect with KDE curves."""
    if "test" not in results:
        return

    data = results["test"]
    max_probs = data["y_probs"].max(axis=1)
    correct = data["y_true"] == data["y_pred"]

    fig, ax = plt.subplots(figsize=(11, 7))
    
    # Plot histograms with smooth KDE lines
    sns.histplot(max_probs[correct], bins=25, kde=True, alpha=0.5, color="#20C997",
                 label=f"Correct ({correct.sum()} samples)", ax=ax, edgecolor="white", linewidth=0.8)
    sns.histplot(max_probs[~correct], bins=25, kde=True, alpha=0.5, color="#E64980",
                 label=f"Incorrect ({(~correct).sum()} samples)", ax=ax, edgecolor="white", linewidth=0.8)
                 
    # Calculate and draw medians
    median_correct = np.median(max_probs[correct]) if correct.sum() > 0 else 0
    median_incorrect = np.median(max_probs[~correct]) if (~correct).sum() > 0 else 0
    
    if correct.sum() > 0:
        ax.axvline(median_correct, color="#0E8388", linestyle="--", lw=2,
                   label=f"Median Correct ({median_correct:.3f})")
    if (~correct).sum() > 0:
        ax.axvline(median_incorrect, color="#C2255C", linestyle="--", lw=2,
                   label=f"Median Incorrect ({median_incorrect:.3f})")

    ax.set_xlabel("Prediction Confidence (Max Class Probability)", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_ylabel("Count / Density", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_title("Prediction Confidence Distribution — Correct vs Incorrect",
                 fontsize=15, fontweight="bold", pad=15, color="#2D3748")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95, facecolor="#FAF9F6", edgecolor="#E2E8F0")
    ax.grid(True, linestyle="--", alpha=0.5)
    
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color("#CCCCCC")
        
    plt.tight_layout()

    fig.savefig(OUTPUT_DIR / "confidence_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] Saved: confidence_distribution.png")


plot_confidence_distribution()


# ---- 8j. Misclassified Samples Grid ----------------------------------------
def plot_misclassified_samples():
    """Show a grid of misclassified test images in a structured card layout."""
    if "test" not in results:
        return

    data = results["test"]
    wrong_idx = np.where(data["y_true"] != data["y_pred"])[0]
    if len(wrong_idx) == 0:
        print("  [INFO] No misclassified test samples! Perfect accuracy.")
        return

    n_show = min(16, len(wrong_idx))
    selected = wrong_idx[:n_show]

    cols = 4
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes.flatten()

    for i, idx in enumerate(selected):
        ax = axes[i]
        img = Image.open(data["paths"][idx]).convert("RGB").resize((224, 224))
        ax.imshow(img)
        
        true_cls = DISEASE_NAMES[data["y_true"][idx]]
        pred_cls = DISEASE_NAMES[data["y_pred"][idx]]
        conf = data["y_probs"][idx].max()
        
        ax.axis("off")
        title_text = f"True: {true_cls}\nPred: {pred_cls} ({conf:.2f})"
        ax.set_title(title_text, fontsize=9.5, color="#D62728", fontweight="bold", pad=8)
        
        # Red border around misclassified cards
        from matplotlib.patches import Rectangle
        rect = Rectangle((0, 0), 223, 223, linewidth=3.5, edgecolor="#E64980", facecolor="none")
        ax.add_patch(rect)

    for i in range(n_show, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Misclassified Test Samples (Error Cases)", fontsize=16, fontweight="bold", y=0.98, color="#2D3748")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUTPUT_DIR / "misclassified_samples.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved: misclassified_samples.png ({n_show} samples shown)")


plot_misclassified_samples()


# ========================= 9. GRAD-CAM HEATMAPS =============================
print("\n" + "=" * 70)
print("  GENERATING GRAD-CAM HEATMAPS")
print("=" * 70)


class GradCAM:
    """Grad-CAM implementation for the HybridQuantumResNet."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """Generate Grad-CAM heatmap."""
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # GAP
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, target_class, output


def generate_gradcam_heatmaps(n_samples=12):
    """Generate Grad-CAM heatmaps for sample test images with clean borders and cards."""
    if "test" not in results:
        print("  [WARN] No test results for Grad-CAM.")
        return

    # Target the last conv layer (layer2 contains layer3 and layer4 of ResNet)
    target_layer = model.layer2[-1]  # last block of layer4

    gradcam = GradCAM(model, target_layer)

    data = results["test"]
    n_show = min(n_samples, len(data["y_true"]))

    # Select diverse samples (one per class if possible)
    selected_indices = []
    for cls in range(NUM_CLASSES):
        cls_indices = np.where(data["y_true"] == cls)[0]
        if len(cls_indices) > 0:
            selected_indices.append(cls_indices[0])
            
    remaining = [i for i in range(len(data["y_true"])) if i not in selected_indices]
    while len(selected_indices) < n_show and remaining:
        selected_indices.append(remaining.pop(0))

    cols = 4
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols * 2, figsize=(4.5 * cols * 2, 4.5 * rows))

    for i, idx in enumerate(selected_indices[:n_show]):
        row = i // cols
        col = i % cols

        img_path = data["paths"][idx]
        img_pil = Image.open(img_path).convert("RGB").resize((224, 224))
        img_np = np.array(img_pil)

        input_tensor = imagenet_tf(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)
        input_tensor.requires_grad_(True)

        try:
            cam, pred_class, logits = gradcam.generate(input_tensor)

            import cv2
            cam_resized = cv2.resize(cam, (224, 224))
            heatmap = plt.cm.jet(cam_resized)[:, :, :3]
            heatmap = (heatmap * 255).astype(np.uint8)

            overlay = (0.5 * img_np + 0.5 * heatmap).astype(np.uint8)

            true_cls = data["y_true"][idx]
            pred_cls_name = DISEASE_NAMES[pred_class]
            true_cls_name = DISEASE_NAMES[true_cls]
            conf = F.softmax(logits, dim=1)[0, pred_class].item()

            # Original image
            ax_orig = axes[row, col * 2] if rows > 1 else axes[col * 2]
            ax_orig.imshow(img_np)
            ax_orig.set_title(f"True: {true_cls_name}", fontsize=9, fontweight="bold", color="#2D3748")
            ax_orig.axis("off")
            
            from matplotlib.patches import Rectangle
            rect_orig = Rectangle((0, 0), 223, 223, linewidth=2, edgecolor="#CCCCCC", facecolor="none")
            ax_orig.add_patch(rect_orig)

            # Grad-CAM overlay
            ax_cam = axes[row, col * 2 + 1] if rows > 1 else axes[col * 2 + 1]
            ax_cam.imshow(overlay)
            
            is_correct = true_cls == pred_class
            text_color = "#20C997" if is_correct else "#E64980"
            ax_cam.set_title(f"Pred: {pred_cls_name}\n(Conf: {conf:.2f})",
                            fontsize=9, fontweight="bold",
                            color=text_color)
            ax_cam.axis("off")
            
            rect_cam = Rectangle((0, 0), 223, 223, linewidth=2.5, edgecolor=text_color, facecolor="none")
            ax_cam.add_patch(rect_cam)

        except Exception as e:
            ax_orig = axes[row, col * 2] if rows > 1 else axes[col * 2]
            ax_cam = axes[row, col * 2 + 1] if rows > 1 else axes[col * 2 + 1]
            ax_orig.axis("off")
            ax_cam.axis("off")
            ax_cam.set_title(f"Error: {str(e)[:30]}", fontsize=8, color="#E64980")

    total_axes = rows * cols * 2
    for i in range(n_show * 2, total_axes):
        r = i // (cols * 2)
        c = i % (cols * 2)
        ax = axes[r, c] if rows > 1 else axes[c]
        ax.axis("off")

    plt.suptitle("Grad-CAM Activation Heatmaps (Visual Explanation)",
                 fontsize=16, fontweight="bold", y=0.98, color="#2D3748")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUTPUT_DIR / "gradcam_heatmaps.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved: gradcam_heatmaps.png ({n_show} samples)")


try:
    import cv2
    generate_gradcam_heatmaps(n_samples=12)
except ImportError:
    print("  [WARN] opencv (cv2) not installed - using PIL fallback for Grad-CAM.")
except Exception as e:
    print(f"  [WARN] Grad-CAM failed: {e}")


# ========================= 10. PER-CLASS METRICS TABLE =======================
def save_per_class_table():
    """Save a detailed per-class metrics table."""
    if "test" not in results:
        return

    data = results["test"]
    mcm = multilabel_confusion_matrix(data["y_true"], data["y_pred"], labels=range(NUM_CLASSES))

    rows = []
    for i in range(NUM_CLASSES):
        tn, fp, fn, tp = mcm[i].ravel()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_val = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f2_val = 5 * prec * rec / (4 * prec + rec) if (4 * prec + rec) > 0 else 0
        support = tp + fn

        rows.append({
            "Class": DISEASE_NAMES[i],
            "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
            "Precision": prec, "Recall": rec, "Specificity": spec,
            "F1": f1_val, "F2": f2_val, "Support": int(support),
        })

    # Print table
    print(f"\n{'-' * 110}")
    print(f"  DETAILED PER-CLASS METRICS TABLE (TEST SET)")
    print(f"{'-' * 110}")
    header = f"  {'Class':25s} | {'TP':>4s} | {'TN':>4s} | {'FP':>4s} | {'FN':>4s} | {'Prec':>7s} | {'Recall':>7s} | {'Spec':>7s} | {'F1':>7s} | {'F2':>7s} | {'Supp':>5s}"
    print(header)
    print(f"  {'-' * 105}")
    for r in rows:
        print(f"  {r['Class']:25s} | {r['TP']:4d} | {r['TN']:4d} | {r['FP']:4d} | {r['FN']:4d} | "
              f"{r['Precision']:7.4f} | {r['Recall']:7.4f} | {r['Specificity']:7.4f} | "
              f"{r['F1']:7.4f} | {r['F2']:7.4f} | {r['Support']:5d}")

    # Save to file
    import csv
    with open(OUTPUT_DIR / "per_class_detailed_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  [INFO] Saved: per_class_detailed_metrics.csv")


save_per_class_table()


# ========================= 11. SUMMARY ======================================
print("\n" + "=" * 70)
print("  EVALUATION COMPLETE")
print("=" * 70)
print(f"\n  All outputs saved to: {OUTPUT_DIR}")
print(f"\n  Generated files:")

for f in sorted(OUTPUT_DIR.iterdir()):
    size_kb = f.stat().st_size / 1024
    print(f"    * {f.name:45s} ({size_kb:.1f} KB)")

print(f"\n{'=' * 70}")
