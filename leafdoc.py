
import os
import math
import random
import json
import time
import warnings
import traceback

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import cv2
import timm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from rich.console import Console
console = Console()


# ---------------------------------------
# CONFIG - tune here for your machine
# ---------------------------------------
class CFG:
    seed = 42

    # Root folder that contains "train/" and "valid/" folders used in labels.csv
    data_root = r"C:/Users/vinod/Downloads/archive/New Plant Diseases Dataset(Augmented)"

    # Path to csv created earlier (relative to project root)
    csv_path = "data/labels.csv"

    # where to save outputs
    out_dir = "runs"

    # Model / data
    backbone = "convnext_tiny"   # efficient + accurate for small GPUs (timm)
    img_size = 320               # smaller to fit 4GB VRAM
    batch_size = 8               # per-step batch size
    gradient_accumulation_steps = 2  # effective batch_size = batch_size * accum_steps
    epochs = 20

    # optimizer / schedule
    lr = 3e-4
    weight_decay = 1e-4

    # DataLoader
    num_workers = 4             # try 4; if you get worker errors reduce to 2 or 0
    pin_memory = True

    # validation split fallback (if csv does not contain valid rows)
    val_size = 0.15

    # normalization (ImageNet)
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # safety
    max_grad_norm = 5.0  # gradient clipping to avoid NaNs


# ---------------------------------------
# Repro / seed
# ---------------------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(CFG.seed)


# ---------------------------------------
# Utility: compute simple interpretable features
# ---------------------------------------
def estimate_features(img_rgb_uint8):
    # Input: HxWx3 uint8 RGB
    try:
        img_bgr = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR)
        B, G, R = cv2.split(img_bgr.astype("float32"))
        rg = np.abs(R - G)
        yb = np.abs(0.5 * (R + G) - B)
        colorfulness = (np.std(rg) + np.std(yb)) / 100.0
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        # the HSV bounds here are a heuristic for green-ish leaf area
        mask = cv2.inRange(hsv, (25, 40, 20), (85, 255, 255))
        leaf_area = mask.mean() / 255.0
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_damage = edges.mean() / 255.0
        return np.clip([colorfulness, leaf_area, edge_damage], 0.0, 1.0).astype("float32")
    except Exception:
        # fallback safe features
        return np.array([0.0, 0.0, 0.0], dtype="float32")


# ---------------------------------------
# Dataset class
# ---------------------------------------
class LeafDataset(Dataset):
    def __init__(self, df, root_dir, transforms=None):
        """
        df: dataframe with columns ['image', 'disease', 'quality'] where 'image' is like 'train/ClassName/xxx.jpg'
        root_dir: path that when joined with df['image'] yields full path to image file
        transforms: albumentations transform
        """
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, rel_path):
        # allow / and \ in CSV and strip leading ./ if present
        p = str(rel_path).lstrip("./")
        p = p.replace("/", os.sep).replace("\\", os.sep)
        return os.path.join(self.root_dir, p)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = str(row['image'])
        img_path = self._resolve_path(rel_path)

        if not os.path.isfile(img_path):
            # raise here to catch problems early when building dataset; filter_missing_paths should avoid this
            raise FileNotFoundError(f"Missing image: {img_path}")

        # load
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed reading image: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # compute small features on a resized copy (deterministic)
        small = cv2.resize(img_rgb, (CFG.img_size, CFG.img_size), interpolation=cv2.INTER_AREA)
        feats_np = estimate_features(small)
        feats = torch.tensor(feats_np, dtype=torch.float32)

        # transforms -> ToTensorV2 yields float32 tensor in CxHxW
        if self.transforms is not None:
            out = self.transforms(image=img_rgb)
            img_tensor = out['image']  # torch tensor float32, normalized
        else:
            # fallback: convert to tensor / normalize manually
            img_float = img_rgb.astype("float32") / 255.0
            img_tensor = torch.from_numpy(img_float.transpose(2, 0, 1)).float()

        # disease index (int)
        # allow either int or string - assume already encoded in df
        disease = row['disease']
        try:
            disease_t = torch.tensor(int(disease), dtype=torch.long)
        except Exception:
            # fallback: if somehow string present, attempt cast
            disease_t = torch.tensor(int(float(disease)), dtype=torch.long)

        # quality: if provided numeric (0..100) or left blank, make a proxy 0..1 value
        q_raw = row.get('quality', "")
        try:
            if pd.isna(q_raw) or q_raw in (None, ""):
                # heuristics: if disease_name exists and equals 'healthy' -> high quality else medium
                q_val = 0.9 if str(row.get('disease_name', "")).strip().lower() == "healthy" else 0.4
            else:
                q_val = float(q_raw) / 100.0
                # clamp
                q_val = float(np.clip(q_val, 0.0, 1.0))
        except Exception:
            q_val = 0.9 if str(row.get('disease_name', "")).strip().lower() == "healthy" else 0.4
        q_t = torch.tensor([q_val], dtype=torch.float32)

        return img_tensor, disease_t, q_t, feats


# ---------------------------------------
# Model: backbone + heads
# ---------------------------------------
class LeafNet(nn.Module):
    def __init__(self, backbone_name, num_classes):
        super().__init__()
        # create backbone with num_classes=0 to get features
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0, global_pool="avg")
        feat_dim = self.backbone.num_features
        self.disease_head = nn.Linear(feat_dim, num_classes)
        self.quality_head = nn.Sequential(nn.Linear(feat_dim, 1), nn.Sigmoid())
        self.feature_head = nn.Sequential(nn.Linear(feat_dim, 3), nn.Sigmoid())

    def forward(self, x):
        # x: float32 tensor BxCxHxW
        f = self.backbone(x)
        d = self.disease_head(f)
        q = self.quality_head(f)
        feat = self.feature_head(f)
        return d, q, feat


# ---------------------------------------
# Transforms
# ---------------------------------------
def get_transforms(img_size=CFG.img_size):
    train_tf = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=20, p=0.5),
        A.Normalize(mean=CFG.mean, std=CFG.std),
        ToTensorV2()
    ])
    val_tf = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=CFG.mean, std=CFG.std),
        ToTensorV2()
    ])
    return train_tf, val_tf


# ---------------------------------------
# Helpers: remove missing files from CSV (so DataLoader won't crash)
# ---------------------------------------
def filter_missing_paths(df, root_dir):
    kept = []
    missed = 0
    for i, row in df.iterrows():
        try:
            p = str(row['image']).lstrip("./").replace("/", os.sep).replace("\\", os.sep)
            p_full = os.path.join(root_dir, p)
            if os.path.isfile(p_full):
                kept.append(row)
            else:
                missed += 1
        except Exception:
            missed += 1

    if missed:
        console.print(f"[yellow]⚠️ Warning: {missed} missing images were removed from the CSV and won't be used.[/yellow]")
    if len(kept) == 0:
        raise RuntimeError("No images found after filtering; check your CSV paths and data_root.")
    newdf = pd.DataFrame(kept).reset_index(drop=True)
    return newdf


# ---------------------------------------
# Train loop
# ---------------------------------------
def train():
    device = torch.device(CFG.device)
    console.print(f"[bold cyan]Device:[/bold cyan] {device}")

    # speedups
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        # set high precision matmul if available (PyTorch 2.x)
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

    # read csv 
    if not os.path.isfile(CFG.csv_path):
        raise RuntimeError(f"CSV not found: {CFG.csv_path}")
    df = pd.read_csv(CFG.csv_path)

    if 'image' not in df.columns or 'disease' not in df.columns:
        raise RuntimeError("labels.csv must contain at least 'image' and 'disease' columns")

    # Keep original disease name column for heuristics (ensure string)
    df['disease_name'] = df['disease'].astype(str)

    # If labels.csv has class names in 'disease' column, encode them
    le = LabelEncoder()
    try:
        df['disease'] = le.fit_transform(df['disease_name'].values)
    except Exception as e:
        console.print("[red]Failed to label-encode disease names:[/red]", e)
        raise

    classes = list(map(str, le.classes_.tolist()))
    num_classes = len(classes)
    console.print(f"[green]Classes ({num_classes}):[/green] {classes}")

    # Filter missing files (saves DataLoader worker crashes)
    df = filter_missing_paths(df, CFG.data_root)

    # Detect explicit valid/ train markers robustly
    # Accept paths starting with "valid/" "valid\" or containing "/valid/" etc.
    img_paths = df['image'].astype(str)
    is_valid = img_paths.str.match(r'^(valid[/\\]|.*[/\\]valid[/\\].*|valid/|valid\\)', na=False)
    if img_paths[is_valid].shape[0] > 0:
        tr_df = df[~is_valid].reset_index(drop=True)
        va_df = df[is_valid].reset_index(drop=True)
        console.print(f"[cyan]Using CSV train/valid markers:[/cyan] train={len(tr_df)} valid={len(va_df)}")
    else:
        # stratified split by disease
        tr_df, va_df = train_test_split(df, test_size=CFG.val_size, stratify=df['disease'], random_state=CFG.seed)
        console.print(f"[cyan]Split dataset:[/cyan] train={len(tr_df)} valid={len(va_df)}")

    # store mapping for inference
    os.makedirs(CFG.out_dir, exist_ok=True)
    classes_json_path = os.path.join(CFG.out_dir, "classes.json")
    with open(classes_json_path, "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2, ensure_ascii=False)

    # transforms + datasets
    tr_tf, va_tf = get_transforms(CFG.img_size)
    tr_ds = LeafDataset(tr_df, CFG.data_root, transforms=tr_tf)
    va_ds = LeafDataset(va_df, CFG.data_root, transforms=va_tf)

    # dataloaders
    # Only set persistent_workers if num_workers > 0 and supported
    persistent_workers = bool(CFG.num_workers > 0)
    # pin_memory only when using CUDA
    pin_memory = bool(CFG.pin_memory and device.type == "cuda")
    tr_dl = DataLoader(tr_ds, batch_size=CFG.batch_size, shuffle=True,
                       num_workers=CFG.num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    va_dl = DataLoader(va_ds, batch_size=CFG.batch_size, shuffle=False,
                       num_workers=CFG.num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)

    # model
    model = LeafNet(CFG.backbone, num_classes).to(device)
    # channels_last helps conv speed on many GPUs
    try:
        model = model.to(memory_format=torch.channels_last)
    except Exception:
        pass

    # optimizer & losses
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    # mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    best_val_loss = float("inf")
    last_ckpt_path = os.path.join(CFG.out_dir, "last.pt")
    best_ckpt_path = os.path.join(CFG.out_dir, "best.pt")

    try:
        # training loop
        for epoch in range(1, CFG.epochs + 1):
            model.train()
            running_loss = 0.0
            iters = 0

            t0 = time.time()
            opt.zero_grad(set_to_none=True)

            for step, (imgs, dis, q, feats) in enumerate(tr_dl, start=1):
                # ensure float32 and channels_last for speed
                imgs = imgs.to(device=device, non_blocking=True)
                # convert to channels_last if model is channels_last
                try:
                    imgs = imgs.contiguous(memory_format=torch.channels_last)
                except Exception:
                    pass
                dis = dis.to(device=device, non_blocking=True)
                q = q.to(device=device, non_blocking=True)
                feats = feats.to(device=device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    d_logits, q_out, f_out = model(imgs)
                    loss_d = ce(d_logits, dis)
                    loss_q = mse(q_out, q)
                    loss_f = mse(f_out, feats)
                    loss = loss_d + 0.3 * loss_q + 0.2 * loss_f
                    loss = loss / CFG.gradient_accumulation_steps

                # guard against non-finite loss
                loss_val = float(loss.item())
                if not math.isfinite(loss_val):
                    console.print(f"[red]Non-finite loss detected at step {step} (loss={loss_val}). Skipping this batch.[/red]")
                    opt.zero_grad(set_to_none=True)
                    continue

                scaler.scale(loss).backward()

                # gradient accumulation step
                if (step % CFG.gradient_accumulation_steps) == 0:
                    # gradient clipping (unscale -> clip -> step)
                    try:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
                    except Exception:
                        pass

                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)

                # accumulate running_loss in original per-sample units
                running_loss += loss_val * imgs.size(0) * CFG.gradient_accumulation_steps
                iters += imgs.size(0)

            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for imgs, dis, q, feats in va_dl:
                    imgs = imgs.to(device=device, non_blocking=True)
                    try:
                        imgs = imgs.contiguous(memory_format=torch.channels_last)
                    except Exception:
                        pass
                    dis = dis.to(device=device, non_blocking=True)
                    q = q.to(device=device, non_blocking=True)
                    feats = feats.to(device=device, non_blocking=True)

                    with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                        d_logits, q_out, f_out = model(imgs)
                        loss_d = ce(d_logits, dis)
                        loss_q = mse(q_out, q)
                        loss_f = mse(f_out, feats)
                        loss = loss_d + 0.3 * loss_q + 0.2 * loss_f

                    # guard
                    loss_val = float(loss.item())
                    if not math.isfinite(loss_val):
                        console.print("[red]Non-finite validation loss detected. Replacing with large value.[/red]")
                        loss_val = 1e3

                    val_loss += loss_val * imgs.size(0)
                    correct += (d_logits.argmax(1) == dis).sum().item()
                    total += imgs.size(0)

            train_loss = running_loss / max(1, len(tr_ds))
            val_loss = val_loss / max(1, len(va_ds))
            acc = correct / max(1, total) if total > 0 else 0.0

            t1 = time.time()
            console.print(f"[yellow]Epoch {epoch}/{CFG.epochs}[/yellow] | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={acc:.3f} | time={(t1 - t0):.1f}s")

            # save last checkpoint each epoch (state_dict + classes list)
            ckpt = {
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "cfg": {
                    "backbone": CFG.backbone,
                    "img_size": CFG.img_size
                },
                "epoch": epoch
            }
            torch.save(ckpt, last_ckpt_path)

            # save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(ckpt, best_ckpt_path)
                console.print("[green]✅ Saved best.pt[/green]")

    except KeyboardInterrupt:
        console.print("[red]Training interrupted by user (KeyboardInterrupt).[/red]")
        traceback.print_exc()
        # save last state if possible
        try:
            torch.save({"model_state_dict": model.state_dict(), "classes": classes}, last_ckpt_path)
            console.print(f"[yellow]Saved partial checkpoint to {last_ckpt_path}[/yellow]")
        except Exception:
            pass
    except Exception:
        console.print("[red]Exception during training:[/red]")
        traceback.print_exc()
        try:
            torch.save({"model_state_dict": model.state_dict(), "classes": classes}, last_ckpt_path)
            console.print(f"[yellow]Saved partial checkpoint to {last_ckpt_path}[/yellow]")
        except Exception:
            pass

    console.print("[bold green]Training finished[/bold green]")


if __name__ == "__main__":
    os.makedirs(CFG.out_dir, exist_ok=True)
    warnings.filterwarnings("ignore")
    train()
