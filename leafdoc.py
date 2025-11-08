import os, random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

import timm
from rich.console import Console
console = Console()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ============================
# CONFIG (Optimized for 4GB VRAM)
# ============================
class CFG:
    seed = 42
    img_size = 384
    batch_size = 16
    epochs = 20
    lr = 3e-4
    weight_decay = 1e-4
    backbone = "convnext_tiny"
    val_size = 0.15
    num_workers = 2

    data_dir = "data"
    img_dir = "data/images"
    csv_path = "data/labels.csv"
    out_dir = "runs"

    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)


# ============================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(CFG.seed)


# ============================
# Feature extractor (Normalized: 0..1)
# ============================
def estimate_features(img_rgb_uint8):
    img_bgr = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR)

    B, G, R = cv2.split(img_bgr.astype("float32"))
    rg = np.abs(R - G)
    yb = np.abs(0.5*(R + G) - B)
    c = (np.std(rg) + np.std(yb)) / 100.0

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (25, 40, 20), (85, 255, 255))
    la = mask.mean() / 255.0

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    ed = edges.mean() / 255.0

    return np.clip([c, la, ed], 0, 1).astype("float32")


# ============================
class LeafDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.img_dir, row["image"])
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"❌ Missing image: {img_path}")

        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        small = cv2.resize(img_rgb, (CFG.img_size, CFG.img_size))
        feat = torch.tensor(estimate_features(small), dtype=torch.float32)

        img = self.transform(image=img_rgb)["image"]

        disease = torch.tensor(int(row["disease"]), dtype=torch.long)

        is_healthy = (row["disease_name"].lower() == "healthy")
        quality = torch.tensor([0.90 if is_healthy else 0.40], dtype=torch.float32)

        return img, disease, quality, feat


# ============================
class LeafNet(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        feat = self.backbone.num_features

        self.disease_head = nn.Linear(feat, num_classes)
        self.quality_head = nn.Sequential(nn.Linear(feat, 1), nn.Sigmoid())
        self.feature_head = nn.Sequential(nn.Linear(feat, 3), nn.Sigmoid())

    def forward(self, x):
        f = self.backbone(x)
        return self.disease_head(f), self.quality_head(f), self.feature_head(f)


# ============================
def get_transforms():
    train_tf = A.Compose([
        A.Resize(CFG.img_size, CFG.img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=20, p=0.6),
        A.Normalize(mean=CFG.mean, std=CFG.std),
        ToTensorV2(),
    ])
    val_tf = A.Compose([
        A.Resize(CFG.img_size, CFG.img_size),
        A.Normalize(mean=CFG.mean, std=CFG.std),
        ToTensorV2(),
    ])
    return train_tf, val_tf


# ============================
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[bold green]Using GPU:[/bold green] {device}")

    df = pd.read_csv(CFG.csv_path)
    df["disease_name"] = df["disease"]

    le = LabelEncoder()
    df["disease"] = le.fit_transform(df["disease_name"])
    num_classes = len(le.classes_)
    console.print(f"[cyan]Classes:[/cyan] {list(le.classes_)}")

    tr_df, va_df = train_test_split(df, test_size=CFG.val_size,
                                    stratify=df["disease"], random_state=CFG.seed)

    tr_tf, va_tf = get_transforms()
    tr_ds = LeafDataset(tr_df, CFG.img_dir, tr_tf)
    va_ds = LeafDataset(va_df, CFG.img_dir, va_tf)

    tr_dl = DataLoader(tr_ds, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers)
    va_dl = DataLoader(va_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)

    model = LeafNet(CFG.backbone, num_classes).to(device)

    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

    scaler = torch.amp.GradScaler("cuda")

    best_val = float("inf")

    for epoch in range(1, CFG.epochs + 1):
        model.train()
        running = 0.0

        for imgs, dis, q, feats in tr_dl:
            imgs, dis, q, feats = imgs.to(device), dis.to(device), q.to(device), feats.to(device)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                d_out, q_out, f_out = model(imgs)
                loss = ce(d_out, dis) + 0.3*mse(q_out, q) + 0.2*mse(f_out, feats)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += loss.item()*imgs.size(0)

        model.eval()
        v_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for imgs, dis, q, feats in va_dl:
                imgs, dis, q, feats = imgs.to(device), dis.to(device), q.to(device), feats.to(device)
                d_out, q_out, f_out = model(imgs)
                loss = ce(d_out, dis) + 0.3*mse(q_out, q) + 0.2*mse(f_out, feats)
                v_loss += loss.item()*imgs.size(0)
                correct += (d_out.argmax(1) == dis).sum().item()
                total += imgs.size(0)

        tr_loss = running / len(tr_ds)
        va_loss = v_loss / len(va_ds)
        acc = correct / total

        console.print(f"[yellow]Epoch {epoch}[/yellow] | Train={tr_loss:.4f} | Val={va_loss:.4f} | Acc={acc:.3f}")

        if va_loss < best_val:
            best_val = va_loss
            torch.save({"model": model.state_dict(), "classes": list(le.classes_)}, f"{CFG.out_dir}/best.pt")
            console.print("[green]✅ Saved best.pt[/green]")


if __name__ == "__main__":
    os.makedirs(CFG.out_dir, exist_ok=True)
    train()
