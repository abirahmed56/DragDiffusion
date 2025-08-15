import torch
import numpy as np
from PIL import Image
from drag_pipeline import DragPipeline
import cv2

# Settings
image_path = "abir.jpeg"
model_path = "runwayml/stable-diffusion-v1-5"

# Device and dtype setup
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
dtype = torch.float16 if device.type == "cuda" else torch.float32

# Load image using OpenCV and convert to RGB
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert image to PIL Image
pil_image = Image.fromarray(img_rgb)

# Points (source and target for drag)
source_points = [(888, 734)]
target_points = [(739, 754)]

# Convert points to tensors
source_points = torch.tensor(source_points, dtype=torch.float32, device=device).unsqueeze(0)
target_points = torch.tensor(target_points, dtype=torch.float32, device=device).unsqueeze(0)

# Load pipeline
pipe = DragPipeline.from_pretrained(model_path, torch_dtype=dtype)
pipe = pipe.to(device)

# Define prompt
prompt = "a photo of a person with neutral face expression"

# Convert PIL image to tensor expected by invert (batch, C, H, W), float32/float16, [0,1]
def pil_to_tensor(img):
    img = img.convert("RGB")
    img = np.array(img).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img).unsqueeze(0)
    return tensor.to(device=device, dtype=dtype)

input_tensor = pil_to_tensor(pil_image)

# 1) Invert image to latents using DDIM inversion to preserve identity
print("Inverting image to latents...")
latents = pipe.invert(
    image=input_tensor,
    prompt=prompt,
    num_inference_steps=50,
    guidance_scale=1.0,
    return_intermediates=False,
)

# 2) Run drag editing on inverted latents
print("Running drag editing...")
edited_images = pipe(
    image=None,  # Pass None since we're giving latents directly
    prompt=prompt,
    latents=latents,
    handle_points=source_points,
    target_points=target_points,
    num_iter=40,
    save_intermediate=False,
)

# 3) Convert output to PIL and save
if isinstance(edited_images, torch.Tensor):
    img_tensor = edited_images[0] if edited_images.ndim == 4 else edited_images
    img_tensor = (img_tensor.clamp(-1, 1) + 1) / 2
    img_np = (img_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    result_pil = Image.fromarray(img_np)
else:
    result_pil = edited_images[0]

output_path = "output_preserve_identity.png"
result_pil.save(output_path)
print(f"Saved result to {output_path}")
