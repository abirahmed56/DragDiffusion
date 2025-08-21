# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2023) B-
# ytedance Inc..  
# *************************************************************************

from PIL import Image
import os
import numpy as np
from einops import rearrange
import torch
import torch.nn.functional as F
from torchvision import transforms
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image

from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import AttnProcsLayers, LoraLoaderMixin
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    SlicedAttnAddedKVProcessor,
)
from diffusers.models.lora import LoRALinearLayer, LoRACompatibleLinear
from diffusers.optimization import get_scheduler
from diffusers.training_utils import unet_lora_state_dict
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0")


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation
        return RobertaSeriesModelWithTransformation
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel
        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    max_length = tokenizer_max_length if tokenizer_max_length is not None else tokenizer.model_max_length
    return tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )


def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=False):
    text_input_ids = input_ids.to(text_encoder.device)
    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(text_input_ids, attention_mask=attention_mask)
    return prompt_embeds[0]


def train_lora(image,
               prompt,
               model_path,
               vae_path,
               save_lora_path,
               lora_step,
               lora_lr,
               lora_batch_size,
               lora_rank,
               progress,
               save_interval=-1):

    # initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision='fp16'
    )
    set_seed(0)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )

    # Initialize models
    noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
    text_encoder_cls = import_model_class_from_model_name_or_path(model_path, revision=None)
    text_encoder = text_encoder_cls.from_pretrained(model_path, subfolder="text_encoder", revision=None)

    if vae_path == "default":
        vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", revision=None)
    else:
        vae = AutoencoderKL.from_pretrained(vae_path)

    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", revision=None)

    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path=model_path,
        vae=vae,
        unet=unet,
        text_encoder=text_encoder,
        scheduler=noise_scheduler,
        torch_dtype=torch.float16
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    unet.to(device, dtype=torch.float16)
    vae.to(device, dtype=torch.float16)
    text_encoder.to(device, dtype=torch.float16)

    # ---------------------------
    # LoRA layer initialization
    # ---------------------------
    unet_lora_parameters = []
    for attn_processor_name, attn_processor in unet.attn_processors.items():
        attn_module = unet
        for n in attn_processor_name.split(".")[:-1]:
            attn_module = getattr(attn_module, n)

        # Convert standard linear layers to LoRA-compatible
        attn_module.to_q = LoRACompatibleLinear.from_linear(attn_module.to_q, rank=lora_rank)
        attn_module.to_k = LoRACompatibleLinear.from_linear(attn_module.to_k, rank=lora_rank)
        attn_module.to_v = LoRACompatibleLinear.from_linear(attn_module.to_v, rank=lora_rank)

        if isinstance(attn_module.to_out, torch.nn.ModuleList):
            attn_module.to_out[0] = LoRACompatibleLinear.from_linear(attn_module.to_out[0], rank=lora_rank)
        else:
            attn_module.to_out = LoRACompatibleLinear.from_linear(attn_module.to_out, rank=lora_rank)

        # Collect parameters
        unet_lora_parameters.extend(attn_module.to_q.parameters())
        unet_lora_parameters.extend(attn_module.to_k.parameters())
        unet_lora_parameters.extend(attn_module.to_v.parameters())
        if isinstance(attn_module.to_out, torch.nn.ModuleList):
            unet_lora_parameters.extend(attn_module.to_out[0].parameters())
        else:
            unet_lora_parameters.extend(attn_module.to_out.parameters())

        # Handle additional KV projections for special processors
        if isinstance(attn_processor, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)):
            attn_module.add_k_proj = LoRACompatibleLinear.from_linear(attn_module.add_k_proj, rank=lora_rank)
            attn_module.add_v_proj = LoRACompatibleLinear.from_linear(attn_module.add_v_proj, rank=lora_rank)
            unet_lora_parameters.extend(attn_module.add_k_proj.parameters())
            unet_lora_parameters.extend(attn_module.add_v_proj.parameters())

    # ---------------------------
    # Optimizer and scheduler
    # ---------------------------
    optimizer = torch.optim.AdamW(
        unet_lora_parameters,
        lr=lora_lr,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )

    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=lora_step,
        num_cycles=1,
        power=1.0,
    )

    # prepare accelerator
    unet, optimizer, lr_scheduler = accelerator.prepare(unet, optimizer, lr_scheduler)

    # ---------------------------
    # Encode text prompt
    # ---------------------------
    with torch.no_grad():
        text_inputs = tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None)
        text_embedding = encode_prompt(
            text_encoder,
            text_inputs.input_ids,
            text_inputs.attention_mask,
            text_encoder_use_attention_mask=False
        )
        text_embedding = text_embedding.repeat(lora_batch_size, 1, 1)

    # ---------------------------
    # Image transforms
    # ---------------------------
    image_transforms_pil = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomCrop(512),
    ])
    image_transforms_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # ---------------------------
    # Training loop
    # ---------------------------
    for step in progress.tqdm(range(lora_step), desc="training LoRA"):
        unet.train()
        image_batch = []
        for _ in range(lora_batch_size):
            image_transformed = image_transforms_pil(Image.fromarray(image))
            image_transformed = image_transforms_tensor(image_transformed).to(device, dtype=torch.float16)
            image_transformed = image_transformed.unsqueeze(dim=0)
            image_batch.append(image_transformed)
        image_batch = torch.cat(image_batch, dim=0)

        latents_dist = vae.encode(image_batch).latent_dist
        model_input = latents_dist.sample() * vae.config.scaling_factor
        noise = torch.randn_like(model_input)
        bsz, channels, height, width = model_input.shape
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device)
        timesteps = timesteps.long()
        noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

        model_pred = unet(noisy_model_input, timesteps, text_embedding).sample

        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(model_input, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if save_interval > 0 and (step + 1) % save_interval == 0:
            save_lora_path_intermediate = os.path.join(save_lora_path, str(step + 1))
            os.makedirs(save_lora_path_intermediate, exist_ok=True)
            unet_lora_layers = unet_lora_state_dict(unet)
            LoraLoaderMixin.save_lora_weights(
                save_directory=save_lora_path_intermediate,
                unet_lora_layers=unet_lora_layers,
                text_encoder_lora_layers=None,
            )

    # save the trained LoRA
    os.makedirs(save_lora_path, exist_ok=True)
    unet_lora_layers = unet_lora_state_dict(unet)
    LoraLoaderMixin.save_lora_weights(
        save_directory=save_lora_path,
        unet_lora_layers=unet_lora_layers,
        text_encoder_lora_layers=None,
    )

    return
