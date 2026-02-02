import os
from collections import defaultdict

# ----------------------------
# CONFIG
# ----------------------------
FOLDERS = {
    "folder_A": "RS_20_R_60",
    "folder_B": "RS_40_R_20",
    "folder_C": "RS_70_R_30",
    "folder_D": "RS_100_R_40",
}

EXT = ".png"


# ----------------------------
# LOAD FILE NAMES
# ----------------------------
files_per_folder = {}

for name, path in FOLDERS.items():
    files = {
        f for f in os.listdir(path)
        if f.lower().endswith(EXT) and os.path.isfile(os.path.join(path, f))
    }
    files_per_folder[name] = files


# ----------------------------
# GLOBAL SETS
# ----------------------------
all_files = set.union(*files_per_folder.values())
common_files = set.intersection(*files_per_folder.values())


# ----------------------------
# REPORT
# ----------------------------
print("\n" + "=" * 60)
print("PNG FILE NAME DIFFERENCES REPORT")
print("=" * 60)

# Missing files per folder
for folder, files in files_per_folder.items():
    missing = sorted(all_files - files)
    print(f"\n📁 {folder}")
    print("-" * 40)

    if not missing:
        print("✔ No missing files")
    else:
        print(f"❌ Missing {len(missing)} files:")
        for f in missing:
            print(f"   - {f}")

# Common files
print("\n" + "=" * 60)
print(f"✅ Files present in ALL folders ({len(common_files)}):")
print("-" * 60)

for f in sorted(common_files):
    print(f"   - {f}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

for folder, files in files_per_folder.items():
    print(f"{folder}: {len(files)} png files")

print(f"\nTotal unique png files: {len(all_files)}")
print(f"Common to all folders:  {len(common_files)}")
