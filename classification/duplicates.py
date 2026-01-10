import pandas as pd
from pathlib import Path

# Putanje (PROMENI OVO)
csv_path = "data/Duplicates.csv"
images_dir = Path("/Users/katarinakrstin/Downloads/ISIC_2020_Training_JPEG")

# Učitaj CSV
df = pd.read_csv(csv_path)
print(f"Učitano {len(df)} redova iz {csv_path}")

# Lista imena slika koje treba obrisati
to_delete = df["image_name_2"].astype(str).tolist()

deleted_files = []

for image_name in to_delete:
    # pronađi sve fajlove sa tim imenom, bez obzira na ekstenziju
    for file in images_dir.rglob(f"{image_name}.jpg"):
        file.unlink()  # briše fajl
        deleted_files.append(file.name)

print(f"Obrisano ukupno {len(deleted_files)} fajlova.")


# Sada ažuriraj CSV fajl da ukloni te unose
csv_path_to_delete = "data/Training_GroundTruth.csv"
df_gt = pd.read_csv(csv_path_to_delete)

print(f"Pre brisanja: {len(df_gt)} redova")

# KLJUČNA LINIJA
df_gt = df_gt[~df_gt["image_name"].astype(str).isin(to_delete)]

print(f"Posle brisanja: {len(df_gt)} redova")

# SNIMI NAZAD (fizički upis)
df_gt.to_csv(csv_path_to_delete, index=False)