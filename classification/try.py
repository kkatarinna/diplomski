import matplotlib.pyplot as plt
import numpy as np
import torch

data = torch.load("data/isic_monk_dataset.pt")

image_tensor = data[0]["image"]        # Tensor [3, 224, 224]
monk_tone = data[0]["monk_tone"]       # int
image_name = data[0]["image_name"]

print(f"Image name: {image_name}")
print(f"Monk tone: {monk_tone}")

mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

image_denorm = image_tensor * std + mean
image_np = image_denorm.permute(1, 2, 0).cpu().numpy()
image_np = np.clip(image_np, 0, 1)

plt.figure(figsize=(4, 4))
plt.imshow(image_np)
plt.axis("off")
plt.title(data[0]["image_name"])
plt.show()
