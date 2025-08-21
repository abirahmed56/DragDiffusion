import torch
import numpy as np
from PIL import Image
import cv2
from drag_pipeline import DragPipeline
from utils.lora_utils import load_lora  # assuming you have this helper to load LoRA

# -------------------- Settings --------------------
image_path = "abir.jpeg"          # original image
mask_path  = "mask.png"           # mask you drew
model_path = "runwayml/stable-diffusion-v1-5"
lora_path  = "./lora/lora.safetensors"  # pre-trained LoRA

# -------------------- Device and dtype --------------------
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
dtype = torch.float16 if device.type == "cuda" else torch.float32

# -------------------- Load Image --------------------
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pil_image = Image.fromarray(img_rgb)

# -------------------- Load Mask --------------------
mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
if mask_img is None:
    raise FileNotFoundError(f"Could not read {mask_path}")
mask_tensor = torch.from_numpy(mask_img.astype(np.float32)/255.0).unsqueeze(0).unsqueeze(0)
mask_tensor = mask_tensor.to(device=device, dtype=dtype)  # [1,1,H,W]

# -------------------- Points (source -> target) --------------------
source_points = [(888, 734)]
target_points = [(739, 754)]
source_points = torch.tensor(source_points, dtype=torch.float32, device=device).unsqueeze(0)
target_points = torch.tensor(target_points, dtype=torch.float32, device=device).unsqueeze(0)

# -------------------- Load Pipeline --------------------
pipe = DragPipeline.from_pretrained(model_path, torch_dtype=dtype)
pipe = pipe.to(device)

# -------------------- Load LoRA --------------------
if lora_path:
    print(f"Loading LoRA from {lora_path}...")
    pipe.unet = load_lora(pipe.unet, lora_path)  # inject LoRA into UNet
    print("LoRA loaded.")

# -------------------- Define Prompt --------------------
prompt = "a photo of a person with neutral face expression"

# -------------------- Helper: PIL to Tensor --------------------
def pil_to_tensor(img):
    img = img.convert("RGB")
    img = np.array(img).astype(np.float32) / 255.0
    img = np.transpose(img, (2,0,1))
    tensor = torch.from_numpy(img).unsqueeze(0)
    return tensor.to(device=device, dtype=dtype)

input_tensor = pil_to_tensor(pil_image)

# -------------------- 1) Invert Image to Latents --------------------
print("Inverting image to latents...")
latents = pipe.invert(
    image=input_tensor,
    prompt=prompt,
    num_inference_steps=50,
    guidance_scale=1.0,
    return_intermediates=False,
)

# -------------------- 2) Run Drag Editing with Mask and LoRA --------------------
print("Running drag editing...")
edited_images = pipe(
    image=None,  # latents provided
    prompt=prompt,
    latents=latents,
    handle_points=source_points,
    target_points=target_points,
    mask=mask_tensor,  # <-- apply mask here
    num_iter=40,
    save_intermediate=False,
)

# -------------------- 3) Convert Output to PIL --------------------
if isinstance(edited_images, torch.Tensor):
    img_tensor = edited_images[0] if edited_images.ndim==4 else edited_images
    img_tensor = (img_tensor.clamp(-1,1)+1)/2
    img_np = (img_tensor.cpu().permute(1,2,0).numpy()*255).astype(np.uint8)
    result_pil = Image.fromarray(img_np)
else:
    result_pil = edited_images[0]

output_path = "output_drag_lora_mask.png"
result_pil.save(output_path)
print(f"Saved result to {output_path}")
