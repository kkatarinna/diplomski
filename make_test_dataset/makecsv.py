import os
import csv

# Zamenite ovim putanjom do foldera sa slikama
folder_path = "/Users/katarinakrstin/Downloads/archive/BKL"

# Zamenite ovim imenom CSV fajla koji želite da kreirate
csv_file_path = "Test.csv"

# Dobijanje liste svih fajlova u folderu
files = os.listdir(folder_path)

# Filter za slike (možeš dodati ili ukloniti ekstenzije po potrebi)
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
image_files = [f for f in files if f.lower().endswith(image_extensions)]

# Pisanje u CSV fajl
with open(csv_file_path, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['image_name', 'target'])  # header
    for image in image_files:
        writer.writerow([image, 0])

print(f"CSV fajl '{csv_file_path}' je uspešno kreiran sa {len(image_files)} slikama.")


# Putanja do drugog foldera sa slikama
second_folder_path = "/Users/katarinakrstin/Downloads/archive/MEL"


# Dobijanje liste svih fajlova u drugom folderu
files = os.listdir(second_folder_path)

# Filter za slike
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
image_files = [f for f in files if f.lower().endswith(image_extensions)]

# Appendovanje u CSV fajl
with open(csv_file_path, mode='a', newline='') as csv_file:  # 'a' za append
    writer = csv.writer(csv_file)
    for image in image_files:
        writer.writerow([image, 1])

print(f"Dodat CSV sa {len(image_files)} slikama iz foldera '{second_folder_path}', target = 1.")