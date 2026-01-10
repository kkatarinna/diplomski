import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

from models.efficienNet import EfficientNetWithFeatures
from classification.loader import ISICMonkIterableDataset

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

test_dataset = ISICMonkIterableDataset(
    "dataset/isic_monk_dataset"  # test set
)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

model = EfficientNetWithFeatures(
    num_extra_features=1,
    pretrained=False  # VAŽNO
)

model.load_state_dict(
    torch.load("efficientnet_with_tone.pth", map_location=device)
)
model.to(device)
model.eval()

all_targets = []
all_preds = []
all_probs = []

with torch.no_grad():
    for images, tones, targets in test_loader:
        images = images.to(device)
        tones = tones.to(device)
        targets = targets.to(device)

        outputs = model(images, tones)

        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).int()

        all_targets.extend(targets.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())


accuracy = accuracy_score(all_targets, all_preds)
precision = precision_score(all_targets, all_preds, zero_division=0)
recall = recall_score(all_targets, all_preds, zero_division=0)
f1 = f1_score(all_targets, all_preds, zero_division=0)

print(" Evaluation results:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

cm = confusion_matrix(all_targets, all_preds)

tn, fp, fn, tp = cm.ravel()

print("Confusion Matrix:")
print(cm)

print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

# False negative i positive rates
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

print(f"Sensitivity (Recall+): {sensitivity:.4f}")
print(f"Specificity (Recall-): {specificity:.4f}")
print(f"FPR: {fpr:.4f}")
print(f"FNR: {fnr:.4f}")

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Pred 0", "Pred 1"],
    yticklabels=["True 0", "True 1"]
)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
plt.savefig("confusion_matrix.png", dpi=300)

# ROC Curve and AUC
# Izračunavanje ROC curve
fpr, tpr, thresholds = roc_curve(all_targets, all_probs)
auc = roc_auc_score(all_targets, all_probs)

print(f"ROC AUC: {auc:.4f}")

# Plot ROC curve
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc:.4f})')
plt.plot([0,1], [0,1], color='red', linestyle='--')  # random baseline
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
plt.savefig("roc_curve.png", dpi=300)