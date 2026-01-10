import pandas as pd
from pathlib import Path

# -------------------------------
# Configuration
# -------------------------------
INPUT_CSV = "data/Training_GroundTruth.csv"
OUTPUT_CSV = "data/Training_GroundTruth_balanced.csv"

IMAGES_DIR = Path("/Users/katarinakrstin/Downloads/ISIC_2020_Training_JPEG")

MAX_PER_CLASS = 2000
RANDOM_SEED = 42

# -------------------------------
# Load data
# -------------------------------
df = pd.read_csv(INPUT_CSV)

df["target"] = pd.to_numeric(df["target"], errors="coerce")
df["monk_skin_tone"] = pd.to_numeric(df["monk_skin_tone"], errors="coerce")

# -------------------------------
# Downsampling logic
# -------------------------------
balanced_groups = []
removed_rows = []

for tone, group in df.groupby("monk_skin_tone"):
    total_count = len(group)

    if total_count <= MAX_PER_CLASS:
        balanced_groups.append(group)
        continue

    malignant = group[group["target"] == 1]
    benign = group[group["target"] == 0]

    malignant_count = len(malignant)

    if malignant_count > MAX_PER_CLASS:
        print(f"WARNING: monk_skin_tone {tone} has more than {MAX_PER_CLASS} malignant samples!")
        kept = malignant.sample(n=MAX_PER_CLASS, random_state=RANDOM_SEED)
        removed = group.drop(kept.index)
        balanced_groups.append(kept)
        removed_rows.append(removed)
        continue

    benign_to_keep = MAX_PER_CLASS - malignant_count

    benign_sampled = benign.sample(
        n=benign_to_keep,
        random_state=RANDOM_SEED
    )

    benign_removed = benign.drop(benign_sampled.index)

    balanced_group = pd.concat([malignant, benign_sampled])
    balanced_groups.append(balanced_group)
    removed_rows.append(benign_removed)

# -------------------------------
# Combine kept & removed
# -------------------------------
balanced_df = pd.concat(balanced_groups).reset_index(drop=True)
removed_df = pd.concat(removed_rows).reset_index(drop=True)

# -------------------------------
# Save balanced CSV
# -------------------------------
balanced_df.to_csv(OUTPUT_CSV, index=False)

print(f"Balansirani CSV snimljen u: {OUTPUT_CSV}")
print(f"Broj slika za brisanje: {len(removed_df)}")

# -------------------------------
# Delete removed images (ROBUST)
# -------------------------------
deleted = 0

to_delete = removed_df["image_name"].astype(str).str.strip().tolist()

for image_name in to_delete:
    # traži fajl bez obzira gde je i bez obzira na ekstenziju
    for file in IMAGES_DIR.rglob(f"{image_name}*"):
        try:
            file.unlink()
            deleted += 1
        except Exception as e:
            print(f"Ne mogu da obrišem {file}: {e}")

print(f"\nObrisano ukupno {deleted} fajlova.")

# -------------------------------
# Verification
# -------------------------------
print("\nFinal counts per monk_skin_tone:")
print(balanced_df["monk_skin_tone"].value_counts().sort_index())

print("\nMalignant counts per monk_skin_tone (unchanged):")
print(
    balanced_df[balanced_df["target"] == 1]
    .groupby("monk_skin_tone")
    .size()
    .sort_index()
)
