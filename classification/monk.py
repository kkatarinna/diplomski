import os
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from classification import (
    colorparallel,
    brightest_color,
    get_closest_monk_tone,
)

# ------------------ PUTANJE ------------------
csv_path = "mel_images.csv"
images_root = Path("/Users/katarinakrstin/Downloads/MILK10k")  # folder gde su sve slike

# ------------------ UČITAJ CSV ------------------
df = pd.read_csv(csv_path)

# Ako kolona ne postoji – napravi je
if "monk_skin_tone" not in df.columns:
    df["monk_skin_tone"] = pd.NA

# ------------------ OBRADA SLIKA ------------------
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
    image_name = row["image_name"]

    image_path = next(images_root.rglob(f"{image_name}"), None)
    if image_path is None:
        print(f"⚠️ Slika {image_name} nije pronađena, preskačem.")
        continue

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"⚠️ Slika {image_name} nije mogla biti učitana, preskačem.")
        continue

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    dominant_color, top_colors = colorparallel(image_rgb)
    lightest_color = brightest_color(*top_colors[:5])
    monk_index, _ = get_closest_monk_tone(lightest_color)

    df.at[idx, "monk_skin_tone"] = monk_index

# ------------------ SAČUVAJ CSV ------------------
df.to_csv(csv_path, index=False)

print("✅ Monk skin tone uspešno dodat u Training_GroundTruth.csv")
