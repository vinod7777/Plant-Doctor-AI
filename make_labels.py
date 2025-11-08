import os, csv

# 1) If images are inside class folders (Healthy/, Blight/, …), set this:
SOURCE_DATASET = r"C:/Users/vinod/Downloads/mango_dataset"  # <-- class folders here

# 2) Where to write labels.csv
OUTPUT_CSV = r"C:/Users/vinod/Downloads/let-try/Plant-Doctor-AI/data/labels.csv"

rows = []
for disease_folder in os.listdir(SOURCE_DATASET):
    folder_path = os.path.join(SOURCE_DATASET, disease_folder)
    if not os.path.isdir(folder_path):
        continue
    for img_name in os.listdir(folder_path):
        if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            # We’ll look for the file by scanning both flat and subfolders at train time,
            # so only store the file name + disease.
            rows.append([img_name, disease_folder, ""])

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["image", "disease", "quality"])
    w.writerows(rows)

print("✅ labels.csv created successfully!")
print(f"Total images labeled: {len(rows)}")
