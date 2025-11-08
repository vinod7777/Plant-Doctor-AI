import os, torch, cv2, numpy as np
from rich.console import Console
from leafdoc import CFG, LeafNet   # same folder: leafdoc.py named as leafdoc.py

console = Console()

CKPT = os.path.join("runs", "best.pt")
IMG  = os.path.join("data", "images", "your_test.jpg")  # change to any file name

ckpt = torch.load(CKPT, map_location="cpu")
classes = ckpt["classes"]

model = LeafNet(CFG.backbone, num_classes=len(classes))
model.load_state_dict(ckpt["model"])
model.eval()

def preprocess(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (CFG.img_size, CFG.img_size))
    img = img.astype(np.float32)/255.0
    img = (img - np.array(CFG.mean)) / np.array(CFG.std)
    img = torch.tensor(img.transpose(2,0,1)).unsqueeze(0)
    return img

x = preprocess(IMG)
with torch.no_grad():
    d, q, feats = model(x)
    disease = classes[int(d.argmax(1))]
    quality = float(q.squeeze().item()) * 100.0
    cf, area, edge = [float(v) for v in feats.squeeze().tolist()]

console.print(f"Disease: [bold]{disease}[/bold]")
console.print(f"Quality: {quality:.1f} / 100")
console.print(f"Features: colorfulness={cf:.3f}, leaf_area={area:.3f}, edge_damage={edge:.3f}")
