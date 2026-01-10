import torch
from torch.utils.data import DataLoader
import csv
from models.efficienNet import EfficientNetWithFeatures
from classification.loader import ISICMonkIterableDataset
from models.loss import BCEWithToneWeight

dataset = ISICMonkIterableDataset("dataset/isic_monk_dataset")
loader = DataLoader(dataset, batch_size=2)  # može i num_workers>0

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = EfficientNetWithFeatures(num_extra_features=1, pretrained=True)
model.to(device)

tone_weights = [1.0,1.0,1.0,1.0]
criterion = BCEWithToneWeight(tone_weights=tone_weights, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_batches = 2
num_epochs = 5  # primer

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, tones, targets in loader:  # sada loader vraća i labels
        images = images.to(device)
        tones = tones.to(device)
        targets = targets.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        outputs = model(images, tones)

        loss = criterion(outputs, targets, tones)  
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/num_batches:.4f}")


torch.save(model.state_dict(), "efficientnet_with_tone.pth")
print("✅ Model state_dict je sačuvan")

csv_path = "predictions.csv"

model.eval()
with torch.no_grad():
    for images, tones, targets in loader:
        images = images.to(device)
        tones = tones.to(device)
        targets = targets.to(device, dtype=torch.float32)
        preds = model(images, targets)
        predicted_class = (preds > 0.5).int()  # 0 ili 1
        print(preds)
        print(predicted_class)

with torch.no_grad():
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        
        # Header (kolone)
        writer.writerow(["pred", "predicted_class", "target"])

        for images, tones, targets in loader:
            images = images.to(device)
            tones = tones.to(device)
            targets = targets.to(device, dtype=torch.float32)

            preds = model(images, targets)   # shape: (batch_size, 1) ili (batch_size,)
            predicted_class = (preds > 0.5).int()

            # Prebacivanje na CPU i u numpy/listu
            preds_cpu = preds.squeeze().cpu().numpy()
            predicted_cpu = predicted_class.squeeze().cpu().numpy()
            targets_cpu = targets.squeeze().cpu().numpy()

            # Ako je batch_size = 1
            if preds_cpu.ndim == 0:
                writer.writerow([
                    float(preds_cpu),
                    int(predicted_cpu),
                    float(targets_cpu)
                ])
            else:
                for p, pc, t in zip(preds_cpu, predicted_cpu, targets_cpu):
                    writer.writerow([
                        float(p),
                        int(pc),
                        float(t)
                    ])


# model = torch.load("efficientnet_with_tone_full.pth")
# model.to(device)
# model.eval()