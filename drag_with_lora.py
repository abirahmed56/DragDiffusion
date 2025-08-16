# drag_with_lora.py
import os
import torch
import numpy as np
import cv2
from PIL import Image
from drag_pipeline import DragPipeline

# -------- settings --------
image_path  = "abir.jpeg"
model_path  = "runwayml/stable-diffusion-v1-5"
lora_path   = "./lora_out/lora.safetensors"      # from step 1
prompt      = "a photo of a person"              # keep neutral; same as training prompt is fine

# device/dtype
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
dtype  = torch.float16 if device.type == "cuda" else torch.float32

# points (x, y) — handle -> target
source_points = [(888, 734)]
target_points = [(739, 754)]

# -------- helpers --------
def pil_to_tensor(img: Image.Image, device, dtype):
    img = img.convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))              # CHW
    ten = torch.from_numpy(arr).unsqueeze(0)        # BCHW
    return ten.to(device=device, dtype=dtype)

# -------- load image --------
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Could not read {image_path}")
img_rgb   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pil_image = Image.fromarray(img_rgb)

# points -> tensors
handle = torch.tensor(source_points, dtype=torch.float32, device=device).unsqueeze(0)
target = torch.tensor(target_points, dtype=torch.float32, device=device).unsqueeze(0)

# -------- load pipeline --------
pipe = DragPipeline.from_pretrained(model_path, torch_dtype=dtype)
pipe = pipe.to(device)

# Load LoRA (tries both formats used in your repo)
if os.path.isfile(lora_path):
    loaded = False
    try:
        # format used in ui_utils.gen_img/run_drag_gen
        pipe.load_lora_weights(lora_path, weight_name=os.path.basename(lora_path) if lora_path.endswith(".safetensors") else None)
        print("Loaded LoRA via pipe.load_lora_weights")
        loaded = True
    except Exception as e:
        print("load_lora_weights failed:", e)
    if not loaded:
        try:
            pipe.unet.load_attn_procs(os.path.dirname(lora_path) or ".")
            print("Loaded LoRA via unet.load_attn_procs (folder)")
            loaded = True
        except Exception as e:
            print("unet.load_attn_procs failed:", e)
    if not loaded:
        raise RuntimeError("Could not load LoRA. Check lora_path and format.")
else:
    print("WARNING: lora_path not found; proceeding without LoRA.")

# -------- inversion (DDIM) --------
input_tensor = pil_to_tensor(pil_image, device, dtype)
print("Inverting image to latents...")
latents = pipe.invert(
    image=input_tensor,         # BCHW tensor in [0,1]
    prompt=prompt,
    num_inference_steps=50,
    guidance_scale=1.0,
    return_intermediates=False,
)
if isinstance(latents, (list, tuple)):
    latents = latents[0]
print("Latents:", latents.shape)

# -------- drag with LoRA --------
print("Running drag with LoRA...")
edited = pipe(
    image=None,                 # use inverted latents
    prompt=prompt,
    latents=latents,
    handle_points=handle,
    target_points=target,
    num_iter=30,                # 20–40; raise for stronger move
    guidance_scale=1.2,         # keep low to preserve identity
    save_intermediate=False,
)

# -------- save --------
if isinstance(edited, torch.Tensor):
    t = edited[0] if edited.ndim == 4 else edited
    t = (t.clamp(-1, 1) + 1) / 2
    out = (t.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    out_pil = Image.fromarray(out)
else:
    out_pil = edited[0]

out_pil.save("output_lora_drag.png")
print("Saved: output_lora_drag.png")
