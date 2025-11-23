import os
import shutil

DATASET = r"C:/Users/vinod/Downloads/archive/New Plant Diseases Dataset(Augmented)"
OUTPUT = r"C:/Users/vinod/Downloads/let-try/Plant-Doctor-AI/data/images"

os.makedirs(OUTPUT, exist_ok=True)

count = 0
for split in ["train", "valid"]:
    split_path = os.path.join(DATASET, split)

    for disease in os.listdir(split_path):
        dpath = os.path.join(split_path, disease)
        if not os.path.isdir(dpath):
            continue

        for img in os.listdir(dpath):
            src = os.path.join(dpath, img)
            dst = os.path.join(OUTPUT, img)

            try:
                shutil.copy(src, dst)
                count += 1
            except:
                pass

print("✅ All images copied!")
print("Total:", count)
