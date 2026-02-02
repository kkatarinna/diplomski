import os
import torch
from tqdm import tqdm

dataset_dir = "dataset/isic_monk_dataset_test"

chunk_files = sorted([
    f for f in os.listdir(dataset_dir)
    if f.endswith(".pt")
])

print(f"Broj chunk fajlova: {len(chunk_files)}")

total_images = 0
target_0 = 0
target_1 = 0

for chunk_file in tqdm(chunk_files, desc="Reading chunks"):
    chunk_path = os.path.join(dataset_dir, chunk_file)
    chunk = torch.load(chunk_path, map_location="cpu")

    total_images += len(chunk)

    for sample in chunk:
        if sample["target"] == 0:
            target_0 += 1
        elif sample["target"] == 1:
            target_1 += 1

print("=== DATASET STATISTIKA ===")
print(f"Ukupno slika: {total_images}")
print(f"Target = 0: {target_0}")
print(f"Target = 1: {target_1}")

from collections import Counter

monk_counter = Counter()

for chunk_file in chunk_files:
    chunk = torch.load(os.path.join(dataset_dir, chunk_file), map_location="cpu")
    for sample in chunk:
        monk_counter[sample["monk_tone"]] += 1

print("=== MONK TONE DISTRIBUCIJA ===")
for tone, count in sorted(monk_counter.items()):
    print(f"Monk tone {tone}: {count}")
