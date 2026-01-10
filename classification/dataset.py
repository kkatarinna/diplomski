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
import pandas as pd

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
    def __init__(self, image_dir, tensor_cache_path, csv_path, transform=None, rebuild=False, chunk_size=32):
        """
        image_dir: folder sa slikama
        tensor_cache_path: npr. dataset/isic_monk_dataset
        transform: EfficientNet transforms
        rebuild: True -> uvek pravi ponovo
        """
        self.tensor_cache_path = tensor_cache_path
        self.image_dir = image_dir
        self.csv_path = csv_path
        self.transform = transform  
        self.rebuild = rebuild
        self.chunk_size = chunk_size

        if os.path.exists(tensor_cache_path) and not rebuild:
            print("Učitavam Tensor Dataset sa diska...")
            self.data = torch.load(tensor_cache_path)
        else:
            print("Kreiram Tensor Dataset (ovo se radi samo jednom)...")
            self.data = self.build_in_exact_chunks(image_dir=self.image_dir, 
                                                   output_dir=self.tensor_cache_path, 
                                                   transform=self.transform, 
                                                   chunk_size=self.chunk_size, 
                                                   csv_path=self.csv_path)
            print(f"Dataset sačuvan u: {tensor_cache_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return (
            sample["image"],
            torch.tensor(sample["monk_tone"], dtype=torch.float32)
        )
    
    def build_in_exact_chunks(
        self,
        image_dir,
        output_dir,
        transform,
        chunk_size=2,
        csv_path="data/Training_GroundTruth_balanced.csv"
    ):
        os.makedirs(output_dir, exist_ok=True)

        df = pd.read_csv(csv_path)
        label_map = dict(zip(df["image_name"], df["target"]))
        monk_map = dict(zip(df["image_name"], df["monk_skin_tone"]))


        image_names = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith(".jpg") or f.endswith(".png")
        ])

        chunk = []
        chunk_idx = 329

        for image_name in tqdm(image_names, desc="Processing images"):
            image_path = os.path.join(image_dir, image_name)

            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                print(f"Warning: Could not read image {image_path}, skipping.")
                continue

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            # dominant_color, top_colors = colorparallel(image_rgb)
            # lightest_color = brightest_color(*top_colors[:5])
            # monk_index, _ = get_closest_monk_tone(lightest_color)

            if transform:
                image_tensor = transform(image_rgb)
            else:
                image_tensor = torch.tensor(image_rgb).permute(2, 0, 1).float() / 255.0

            # name_only = os.path.splitext(image_name)[0]
            target = int(label_map.get(image_name, 0))  # postavljanje targeta u label
            monk_index = int(monk_map.get(image_name, 0))  # postavljanje monk tona
            # print(f"{image_name} -> Target: {target}, Monk tone: {monk_index}, ")

            chunk.append({
                "image": image_tensor,
                "monk_tone": monk_index,
                "image_name": image_name,
                "target": target
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
            image_dir="/Users/katarinakrstin/Downloads/MILK10k",
            tensor_cache_path="dataset/isic_monk_dataset_mel",
            transform=efficientnet_b0_transforms,
            csv_path="Test.csv",
            rebuild=True,
            chunk_size=32
        )

    # Za učitavanje bez rebuild-a
    # dataset = ISICMonkDataset(
    #     image_dir="data/train_images",   # koristi se samo ako se rebuilduje
    #     tensor_cache_path="data/isic_monk_dataset.pt"
    # )

