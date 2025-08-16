# train_lora_single_image.py
import os
import cv2
import torch
from PIL import Image

# If lora_utils.py is at repo root:
from lora_utils import train_lora
# If it's in a package folder (e.g., dragdiff): from dragdiff.lora_utils import train_lora

# --- Inputs ---
image_path   = "abir.jpeg"                  # your person image
prompt       = "a photo of a person"        # short, neutral; avoid attributes you won't keep later
model_path   = "runwayml/stable-diffusion-v1-5"
vae_path     = "default"                    # or a specific VAE path
out_dir      = "/lora/lora_ckpt"
os.makedirs(out_dir, exist_ok=True)
lora_path    = os.path.join(out_dir, "lora.safetensors")

# --- LoRA training hyperparams (light & fast) ---
lora_step        = 800        # 400–1200 is reasonable for 1–5 images
lora_lr          = 1e-4       # 5e-5 – 2e-4 range
lora_batch_size  = 1
lora_rank        = 8          # 4–16; higher learns more but may overfit

# --- Load the image (as numpy) ---
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Could not read {image_path}")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# --- Train ---
# train_lora expects a numpy HxWxC image (uint8) and writes lora.safetensors to lora_path
print("Starting LoRA training...")
status = train_lora(
    original_image=img,
    prompt=prompt,
    model_path=model_path,
    vae_path=vae_path,
    lora_path=lora_path,
    lora_step=lora_step,
    lora_lr=lora_lr,
    lora_batch_size=lora_batch_size,
    lora_rank=lora_rank,
    progress=None
)
print("LoRA training finished. Saved to:", lora_path)
