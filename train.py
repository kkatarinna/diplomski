import torch
from torch.utils.data import DataLoader
import csv
from tqdm import tqdm
from itertools import islice
from models.efficienNet import EfficientNetWithFeatures
from classification.loader import ISICMonkIterableDataset
from models.loss import BCEWithToneWeight

dataset = ISICMonkIterableDataset("dataset/isic_monk_dataset")
loader = DataLoader(dataset, batch_size=32)  # može i num_workers>0

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = EfficientNetWithFeatures(num_extra_features=1, pretrained=True)
model.to(device)

print("Model je učitan na device:", device)
tone_weights = [1.0, 1.0, 1.0, 1.0, 1.0,
    4.0,
    6.0,
    8.0,
    15.0,
    20.0]
criterion = BCEWithToneWeight(tone_weights=tone_weights, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_batches = 358
num_epochs = 10  # primer

print("Počinje treniranje...")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    num_processed = 0  # broj slika obrađenih u ovoj epohi

    # tqdm loop sa fiksnim brojem batch-eva
    loop = tqdm(islice(loader, num_batches), total=num_batches,
                desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

    for images, tones, targets in loop:
        images = images.to(device)
        tones = tones.to(device)
        targets = targets.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        outputs = model(images, tones)

        loss = criterion(outputs, targets, tones)
        loss.backward()
        optimizer.step()

        # Update running loss
        running_loss += loss.item() * images.size(0)
        num_processed += images.size(0)

        # Update tqdm bar sa trenutnim prosečnim loss-om
        loop.set_postfix(avg_loss=(running_loss / num_processed))

    # Prosečan loss po epohi
    epoch_loss = running_loss / num_processed
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {epoch_loss:.4f}")

# print("Počinje treniranje...")

# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     loop = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

#     for images, tones, targets in loader:  # sada loader vraća i labels
#         images = images.to(device)
#         tones = tones.to(device)
#         targets = targets.to(device, dtype=torch.float32)

#         optimizer.zero_grad()
#         outputs = model(images, tones)

#         loss = criterion(outputs, targets, tones)  
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item() * images.size(0)

#     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/num_batches:.4f}")


torch.save(model.state_dict(), "efficientnet_with_loss.pth")
print("✅ Model state_dict je sačuvan")

# csv_path = "predictions_loss.csv"

# model.eval()
# with torch.no_grad():
#     for images, tones, targets in loader:
#         images = images.to(device)
#         tones = tones.to(device)
#         targets = targets.to(device, dtype=torch.float32)
#         preds = model(images, targets)
#         predicted_class = (preds > 0.5).int()  # 0 ili 1
#         # print(preds)
#         # print(predicted_class)

# with torch.no_grad():
#     with open(csv_path, mode="w", newline="") as f:
#         writer = csv.writer(f)
        
#         # Header (kolone)
#         writer.writerow(["pred", "predicted_class", "target"])

#         for images, tones, targets in loader:
#             images = images.to(device)
#             tones = tones.to(device)
#             targets = targets.to(device, dtype=torch.float32)

#             preds = model(images, targets)   # shape: (batch_size, 1) ili (batch_size,)
#             predicted_class = (preds > 0.5).int()

#             # Prebacivanje na CPU i u numpy/listu
#             preds_cpu = preds.squeeze().cpu().numpy()
#             predicted_cpu = predicted_class.squeeze().cpu().numpy()
#             targets_cpu = targets.squeeze().cpu().numpy()

#             # Ako je batch_size = 1
#             if preds_cpu.ndim == 0:
#                 writer.writerow([
#                     float(preds_cpu),
#                     int(predicted_cpu),
#                     float(targets_cpu)
#                 ])
#             else:
#                 for p, pc, t in zip(preds_cpu, predicted_cpu, targets_cpu):
#                     writer.writerow([
#                         float(p),
#                         int(pc),
#                         float(t)
#                     ])


# model = torch.load("efficientnet_with_tone_full.pth")
# model.to(device)
# model.eval()