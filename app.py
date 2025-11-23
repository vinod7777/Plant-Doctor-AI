# app.py
import gradio as gr
import torch
import numpy as np
import json
import cv2

from leafdoc import CFG, LeafNet

CKPT_PATH = "runs/best.pt"

# ---- Load checkpoint safely ----
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

if "model" in ckpt:
    state_dict = ckpt["model"]
elif "model_state_dict" in ckpt:
    state_dict = ckpt["model_state_dict"]
elif "state_dict" in ckpt:
    state_dict = ckpt["state_dict"]
else:
    raise KeyError(f"❌ ERROR: No model weights found in checkpoint. Available keys: {ckpt.keys()}")

classes = ckpt.get("classes", [])
num_classes = len(classes)

# ---- Init model ----
model = LeafNet(CFG.backbone, num_classes)
model.load_state_dict(state_dict, strict=False)
model.eval()

# Normalization
mean = torch.tensor(CFG.mean).view(1, 3, 1, 1)
std  = torch.tensor(CFG.std).view(1, 3, 1, 1)

@torch.no_grad()
def predict(img_np):

    if img_np is None:
        return "No image", "", {}

    if img_np.dtype != np.uint8:
        img_np = (img_np * 255).astype(np.uint8)

    img_resized = cv2.resize(img_np, (CFG.img_size, CFG.img_size))

    x = torch.from_numpy(img_resized).permute(2,0,1).unsqueeze(0).float() / 255.0
    x = (x - mean) / std

    d_logits, q_out, f_out = model(x)

    # output
    disease_id = int(torch.argmax(d_logits, dim=1).item())
    disease = classes[disease_id] if classes else str(disease_id)

    quality = float(q_out.item())
    feats = f_out.squeeze(0).tolist()

    return (
        disease,
        round(quality * 100, 2),  # convert 0–1 to 0–100
        {
            "colorfulness": round(feats[0], 3),
            "leaf_area": round(feats[1], 3),
            "edge_damage": round(feats[2], 3),
        }
    )

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs=[
        gr.Textbox(label="Disease"),
        gr.Textbox(label="Quality (0-100)"),
        gr.JSON(label="Features"),
    ],
    title="Plant Doctor AI",
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
