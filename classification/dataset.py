import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from classification import (
    colorparallel,
    brightest_color,
    get_closest_monk_tone,
)
from tqdm import tqdm
from torch.utils.data import ConcatDataset

# Transformacije za EfficientNet-B0
efficientnet_b0_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class ISICMonkDataset(Dataset):
    def __init__(self, image_dir, tensor_cache_path, transform=None, rebuild=False):
        """
        image_dir: folder sa slikama
        tensor_cache_path: npr. dataset/isic_monk_dataset.pt
        transform: EfficientNet transforms
        rebuild: True -> uvek pravi ponovo
        """
        self.tensor_cache_path = tensor_cache_path
        self.image_dir = image_dir
        self.transform = transform  
        self.rebuild = rebuild

        if os.path.exists(tensor_cache_path) and not rebuild:
            print("Učitavam Tensor Dataset sa diska...")
            self.data = torch.load(tensor_cache_path)
        else:
            print("Kreiram Tensor Dataset (ovo se radi samo jednom)...")
            output_dir="dataset/chunks"
            self.data = self.build_in_exact_chunks(image_dir=self.image_dir, output_dir=output_dir, transform=self.transform, chunk_size=2)
            print(f"Dataset sačuvan u: {tensor_cache_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return (
            sample["image"],
            torch.tensor(sample["monk_tone"], dtype=torch.float32)
        )
    
    def _build_dataset(self, image_dir, transform):
        samples = []

        image_names = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith(".jpg") or f.endswith(".png")
        ])

        for image_name in tqdm(image_names, desc="Processing images"):
            image_path = os.path.join(image_dir, image_name)

            # Učitavanje slike
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                continue

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            # Ekstrakcija boje kože (TVOJ KOD)
            dominant_color, top_colors = colorparallel(image_rgb)
            lightest_color = brightest_color(*top_colors[:5])
            monk_index, _ = get_closest_monk_tone(lightest_color)

            print(f"{image_name} -> Monk tone: {monk_index}")

            # transform u tensor
            if transform:
                image_tensor = transform(image_rgb)
            else:
                image_tensor = torch.tensor(image_rgb).permute(2, 0, 1).float() / 255.0

            samples.append({
                "image": image_tensor,
                "monk_tone": monk_index,
                "image_name": image_name
            })

        return samples
    
    def build_in_exact_chunks(
        self,
        image_dir,
        output_dir,
        transform,
        chunk_size=2
    ):
        os.makedirs(output_dir, exist_ok=True)

        image_names = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith(".jpg") or f.endswith(".png")
        ])

        chunk = []
        chunk_idx = 0

        for image_name in tqdm(image_names, desc="Processing images"):
            image_path = os.path.join(image_dir, image_name)

            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                continue

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            dominant_color, top_colors = colorparallel(image_rgb)
            lightest_color = brightest_color(*top_colors[:5])
            monk_index, _ = get_closest_monk_tone(lightest_color)

            if transform:
                image_tensor = transform(image_rgb)
            else:
                image_tensor = torch.tensor(image_rgb).permute(2, 0, 1).float() / 255.0

            chunk.append({
                "image": image_tensor,
                "monk_tone": monk_index,
                "image_name": image_name
            })

            if len(chunk) == chunk_size:
                torch.save(
                    chunk,
                    os.path.join(output_dir, f"chunk_{chunk_idx:06d}.pt")
                )
                chunk.clear()
                chunk_idx += 1

        if chunk:
            torch.save(
                chunk,
                os.path.join(output_dir, f"chunk_{chunk_idx:06d}.pt")
            )

if __name__ == "__main__":
    

    dataset = ISICMonkDataset(
            image_dir="data/trening",
            tensor_cache_path="dataset/isic_monk_dataset.pt",
            transform=efficientnet_b0_transforms,
            rebuild=True
        )

    # Za učitavanje bez rebuild-a
    # dataset = ISICMonkDataset(
    #     image_dir="data/train_images",   # koristi se samo ako se rebuilduje
    #     tensor_cache_path="data/isic_monk_dataset.pt"
    # )

