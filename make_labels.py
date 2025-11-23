import os
import csv

# ✅ 1) Base dataset path — where "train" and "valid" folders are located
SOURCE_DATASET = r"C:/Users/vinod/Downloads/archive/New Plant Diseases Dataset(Augmented)"

# ✅ 2) Output CSV file (change path if needed)
OUTPUT_CSV = r"C:/Users/vinod/Downloads/let-try/Plant-Doctor-AI/data/labels.csv"

rows = []

# ✅ Scan both 'train' and 'valid' folders
for split in ["train", "valid"]:
    split_path = os.path.join(SOURCE_DATASET, split)
    if not os.path.exists(split_path):
        print(f"⚠️ Skipping missing folder: {split_path}")
        continue

    # Each subfolder = disease class
    for disease_folder in os.listdir(split_path):
        folder_path = os.path.join(split_path, disease_folder)
        if not os.path.isdir(folder_path):
            continue

        for img_name in os.listdir(folder_path):
            if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                img_full_path = os.path.join(split, disease_folder, img_name)
                rows.append([img_full_path, disease_folder, ""])

# ✅ Write CSV file
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "disease", "quality"])
    writer.writerows(rows)

print("✅ labels.csv created successfully!")
print(f"Total images labeled: {len(rows)}")
