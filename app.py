# app.py
import gradio as gr
import torch
import numpy as np
import json
import cv2

from leafdoc import CFG, LeafNet  # uses the same config/model as training

CKPT = "runs/best.pt"

# ---- load checkpoint ----
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
classes = list(ckpt.get("classes", []))
num_classes = len(classes)

model = LeafNet(CFG.backbone, num_classes)
model.load_state_dict(ckpt["model"], strict=True)
model.eval()

# tensors for normalization (float32)
mean = torch.tensor(CFG.mean, dtype=torch.float32).view(1, 3, 1, 1)
std  = torch.tensor(CFG.std,  dtype=torch.float32).view(1, 3, 1, 1)

@torch.no_grad()
def predict(img_np):
    """
    img_np: RGB numpy array from Gradio (H, W, C), dtype=uint8 or float
    returns: disease, quality, features dict
    """

    # ensure RGB uint8
    if img_np is None:
        return "No image", "", {}
    if img_np.dtype != np.uint8:
        img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)

    # resize to training size
    img_resized = cv2.resize(img_np, (CFG.img_size, CFG.img_size), interpolation=cv2.INTER_LINEAR)

    # to torch float32, normalized
    x = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).to(torch.float32) / 255.0
    x = (x - mean) / std

    # forward
    d_logits, q_out, f_out = model(x)

    # decode
    disease_idx = int(d_logits.argmax(1).item())
    disease = classes[disease_idx] if classes else str(disease_idx)

    quality = float(q_out.squeeze(0).squeeze(0).item())
    feats = f_out.squeeze(0).cpu().numpy().tolist()
    feats = {
        "colorfulness": float(feats[0]),
        "leaf_area": float(feats[1]),
        "edge_damage": float(feats[2]),
    }

    return disease, round(quality, 2), feats

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload a leaf (RGB)"),
    outputs=[gr.Textbox(label="Disease"),
             gr.Textbox(label="Quality (0-100)"),
             gr.JSON(label="Features")],
    title="Leaf Doctor",
    allow_flagging="never"
)

if __name__ == "__main__":
    # 404 on /manifest.json from browser devtools is harmless in some Gradio versions.
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
