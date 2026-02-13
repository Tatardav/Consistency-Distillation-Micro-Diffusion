import os
import csv
import torch
from tqdm.auto import tqdm
from cd_utils import get_sigmas, edm_forward
from torchvision.utils import save_image


@torch.no_grad()
def consistency_sample(
    student_dit, edm_config, text_emb, latent_shape,
    num_steps=4, sigma_min=0.002, sigma_max=80.0, rho=7.0,
    device='cuda', generator=None,
):
    """Multi-step consistency sampling: denoise → re-noise → denoise → ..."""
    sigmas = get_sigmas(num_steps, sigma_min, sigma_max, rho, device=device)
    x = torch.randn(latent_shape, device=device, generator=generator) * sigmas[0]

    for i, sigma in enumerate(sigmas):
        sigma_batch = torch.full((latent_shape[0],), sigma.item(), device=device)
        x_denoised = edm_forward(
            x.float(), sigma_batch, text_emb.float(),
            student_dit.forward_without_cfg, edm_config
        )
        if i < num_steps - 1:
            noise = torch.randn(latent_shape, device=device, generator=generator)
            x = x_denoised + sigmas[i + 1] * noise
        else:
            x = x_denoised

    return x.float()


@torch.no_grad()
def generate_images(student_dit, model, prompts, num_steps=4, seed=None, device='cuda'):
    """Generate images from prompts using the consistency model."""
    generator = torch.Generator(device=device)
    if seed is not None:
        generator = generator.manual_seed(seed)

    out = model.tokenizer.tokenize(prompts)
    input_ids = out['input_ids'].to(device)
    attn_mask = out.get('attention_mask', None)
    if attn_mask is not None:
        attn_mask = attn_mask.to(device)
    text_emb = model.text_encoder.encode(input_ids, attention_mask=attn_mask)[0]

    latent_shape = (len(prompts), model.dit.in_channels, model.latent_res, model.latent_res)

    latents = consistency_sample(
        student_dit, model.edm_config, text_emb, latent_shape,
        num_steps=num_steps,
        sigma_min=model.edm_config.sigma_min,
        sigma_max=model.edm_config.sigma_max,
        rho=model.edm_config.rho,
        device=device, generator=generator,
    )

    latents = latents / model.latent_scale
    dtype = next(model.vae.parameters()).dtype
    images = model.vae.decode(latents.to(dtype)).sample
    images = (images / 2 + 0.5).clamp(0, 1)
    return images.float()


def load_parti_prompts(tsv_path, num_prompts=100):
    prompts = []
    with open(tsv_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            prompts.append(row['Prompt'])
            if len(prompts) >= num_prompts:
                break
    return prompts


def run_parti_prompts_benchmark(
    student_dit, model, tsv_path="PartiPrompts.tsv",
    output_dir="output/parti_prompts", num_steps=4,
    num_prompts=100, batch_size=4, seed=2024, device='cuda',
):
    os.makedirs(output_dir, exist_ok=True)
    prompts = load_parti_prompts(tsv_path, num_prompts)
    results = []

    student_dit.eval()
    for i in tqdm(range(0, len(prompts), batch_size), desc="PartiPrompts"):
        batch_prompts = prompts[i:i + batch_size]
        images = generate_images(
            student_dit, model, batch_prompts,
            num_steps=num_steps, seed=seed + i, device=device
        )
        for j, (prompt, img) in enumerate(zip(batch_prompts, images)):
            img_path = os.path.join(output_dir, f"{i + j:04d}.png")
            save_image(img, img_path)
            results.append((prompt, img_path))

    with open(os.path.join(output_dir, "prompts.txt"), 'w') as f:
        for prompt, img_path in results:
            f.write(f"{os.path.basename(img_path)}\t{prompt}\n")

    print(f"Generated {len(results)} images in {output_dir}")
    return results
