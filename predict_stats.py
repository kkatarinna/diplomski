import torch
import numpy as np
import csv
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score
)
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.efficienNet import EfficientNetWithFeatures
from classification.loader import ISICMonkIterableDataset


# =====================================================
# DEVICE
# =====================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# =====================================================
# DATA
# =====================================================
test_dataset = ISICMonkIterableDataset(
    "dataset/isic_monk_dataset_test"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False
)


# =====================================================
# MODEL
# =====================================================
model = EfficientNetWithFeatures(
    num_extra_features=1,
    pretrained=False
)

model.load_state_dict(
    torch.load("efficientnet_with_tone.pth", map_location=device)
)

model.to(device)
model.eval()


# =====================================================
# STORAGE
# =====================================================
all_targets = []
all_preds = []
all_probs = []
all_tones = []


# =====================================================
# EVALUATION LOOP + CSV
# =====================================================
csv_file = "predictions.csv"

with open(csv_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["target", "predicted", "probability", "monk_tone"])

    with torch.no_grad():
        for images, tones, targets in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            tones = tones.to(device)
            targets = targets.to(device)

            outputs = model(images, tones)
            probs = torch.sigmoid(outputs).squeeze(-1)
            preds = (probs > 0.5).int()

            for t, p, pr, tone in zip(
                targets.cpu().numpy(),
                preds.cpu().numpy(),
                probs.cpu().numpy(),
                tones.cpu().numpy()
            ):
                writer.writerow([int(t), int(p), float(pr), int(tone)])

            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_tones.extend(tones.cpu().numpy())

print(f"Predictions saved to {csv_file}")


# =====================================================
# GLOBAL METRICS
# =====================================================
accuracy = accuracy_score(all_targets, all_preds)
precision = precision_score(all_targets, all_preds, zero_division=0)
recall = recall_score(all_targets, all_preds, zero_division=0)
f1 = f1_score(all_targets, all_preds, zero_division=0)

print("\n=== GLOBAL EVALUATION ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

cm = confusion_matrix(all_targets, all_preds)
tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix:")
print(cm)
print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

# Global confusion matrix plot
plt.figure(figsize=(4,4))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Pred 0", "Pred 1"],
    yticklabels=["True 0", "True 1"]
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Global Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix_global.png", dpi=300)
plt.show()


# =====================================================
# ROC CURVE
# =====================================================
fpr, tpr, _ = roc_curve(all_targets, all_probs)
auc = roc_auc_score(all_targets, all_probs)

print(f"\nROC AUC: {auc:.4f}")

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=300)
plt.show()


# =====================================================
# METRICS PER MONK SKIN TONE
# =====================================================
print("\n=== METRICS PER MONK SKIN TONE (MST) ===")

unique_tones = np.unique(all_tones)
metrics_per_tone = {}

for tone in unique_tones:
    idx = np.where(np.array(all_tones) == tone)[0]

    y_true = np.array(all_targets)[idx]
    y_pred = np.array(all_preds)[idx]
    y_prob = np.array(all_probs)[idx]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm_tone = confusion_matrix(y_true, y_pred)

    metrics_per_tone[tone] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm_tone
    }

    print(f"\n--- Monk Skin Tone {int(tone)} ---")
    print(f"Samples : {len(idx)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm_tone)

    # Plot per-tone confusion matrix
    plt.figure(figsize=(4,4))
    sns.heatmap(
        cm_tone, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix – Monk Tone {int(tone)}")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_monk_{int(tone)}.png", dpi=300)
    plt.show()

# =====================================================
# DEMOGRAPHIC PARITY (FAIRNESS)
# =====================================================
print("\n=== DEMOGRAPHIC PARITY (MONK SKIN TONE 1–10) ===")

demographic_parity = {}

all_tones_np = np.array(all_tones)
all_preds_np = np.array(all_preds)

for tone in sorted(unique_tones):
    idx = np.where(all_tones_np == tone)[0]

    if len(idx) == 0:
        continue

    # P(ŷ = 1 | A = tone)
    dp = all_preds_np[idx].mean()

    demographic_parity[int(tone)] = {
        "dp": dp,
        "samples": len(idx)
    }

    print(
        f"Monk {int(tone):2d} | "
        f"Samples: {len(idx):4d} | "
        f"Demographic Parity P(ŷ=1): {dp:.4f}"
    )

# =====================================================
# DEMOGRAPHIC PARITY GAP
# =====================================================
dp_values = [v["dp"] for v in demographic_parity.values()]

dp_gap = max(dp_values) - min(dp_values)
dp_std = np.std(dp_values)

print("\n=== DEMOGRAPHIC PARITY SUMMARY ===")
print(f"Demographic Parity Gap (max - min): {dp_gap:.4f}")
print(f"Demographic Parity Std Dev       : {dp_std:.4f}")

# =====================================================
# EQUAL OPPORTUNITY (FAIRNESS)
# =====================================================
print("\n=== EQUAL OPPORTUNITY (TPR) – MONK SKIN TONE 1–10 ===")

equal_opportunity = {}

all_targets_np = np.array(all_targets).astype(int).reshape(-1)
all_preds_np = np.array(all_preds).astype(int).reshape(-1)
all_tones_np = np.array(all_tones).astype(int).reshape(-1)

for tone in sorted(unique_tones):
    idx = np.where(all_tones_np == tone)[0]

    if len(idx) == 0:
        continue

    y_true = all_targets_np[idx]
    y_pred = all_preds_np[idx]

    # Samo stvarno pozitivni slučajevi
    positives = y_true == 1

    if positives.sum() == 0:
        tpr = np.nan  # nema smisla računati
    else:
        tpr = (y_pred[positives] == 1).mean()

    equal_opportunity[int(tone)] = {
        "tpr": tpr,
        "positives": positives.sum(),
        "samples": len(idx)
    }

    print(
        f"Monk {int(tone):2d} | "
        f"Samples: {len(idx):4d} | "
        f"Positives: {positives.sum():3d} | "
        f"TPR: {'N/A' if np.isnan(tpr) else f'{tpr:.4f}'}"
    )


# =====================================================
# EQUAL OPPORTUNITY GAP
# =====================================================
tpr_values = [
    v["tpr"] for v in equal_opportunity.values()
    if not np.isnan(v["tpr"])
]

if len(tpr_values) > 0:
    eo_gap = max(tpr_values) - min(tpr_values)
    eo_std = np.std(tpr_values)

    print("\n=== EQUAL OPPORTUNITY SUMMARY ===")
    print(f"Equal Opportunity Gap (max - min): {eo_gap:.4f}")
    print(f"Equal Opportunity Std Dev       : {eo_std:.4f}")
else:
    print("\nNo valid TPR values to compute Equal Opportunity.")
