import torch
import torch.nn.functional as F
from functools import partial


def get_sigmas(num_steps, sigma_min=0.002, sigma_max=80.0, rho=7.0, device='cpu'):
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    sigmas = (
        sigma_max ** (1 / rho) +
        step_indices / max(num_steps - 1, 1) *
        (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    return sigmas.float()


def edm_forward(x, sigma, y, model_fwd, edm_config):
    sigma_4d = sigma.to(x.dtype).reshape(-1, 1, 1, 1)
    c_skip = edm_config.sigma_data ** 2 / (sigma_4d ** 2 + edm_config.sigma_data ** 2)
    c_out = sigma_4d * edm_config.sigma_data / (sigma_4d ** 2 + edm_config.sigma_data ** 2).sqrt()
    c_in = 1 / (edm_config.sigma_data ** 2 + sigma_4d ** 2).sqrt()
    c_noise = sigma_4d.log() / 4

    F_x = model_fwd(
        (c_in * x).to(x.dtype),
        c_noise.flatten(),
        y,
        mask_ratio=0
    )['sample']

    D_x = c_skip * x + c_out * F_x.to(x.dtype)
    return D_x


def cd_loss(
    student_dit, teacher_dit, edm_config,
    latents, text_emb,
    sigmas, guidance_scale,
    loss_type='huber', huber_c=0.001,
    weight_dtype=torch.float16
):
    """
    Consistency Distillation loss (EDM formulation).

    Student learns to map (x_σ, σ) → x_0, matching teacher's CFG-guided ODE.
    """
    bsz = latents.shape[0]
    device = latents.device
    N = len(sigmas)

    # random sigma pair (σ_s > σ_t)
    index = torch.randint(0, N - 1, (bsz,), device=device)
    sigma_s = sigmas[index]
    sigma_t = sigmas[index + 1]

    # add noise
    noise = torch.randn_like(latents)
    x_s = latents + sigma_s.view(-1, 1, 1, 1) * noise

    # student prediction (no CFG — it learns the CFG'd result directly)
    student_pred = edm_forward(
        x_s, sigma_s, text_emb,
        student_dit.forward_without_cfg,
        edm_config
    )

    # teacher prediction (with CFG) + Euler ODE step
    with torch.no_grad():
        teacher_denoised = edm_forward(
            x_s.to(weight_dtype),
            sigma_s.to(weight_dtype),
            text_emb.to(weight_dtype),
            partial(teacher_dit.forward, cfg=guidance_scale),
            edm_config
        )

        # Euler step: dx/dσ = (x - D(x;σ)) / σ
        d = (x_s.to(weight_dtype) - teacher_denoised) / sigma_s.view(-1, 1, 1, 1).to(weight_dtype)
        x_t = x_s.to(weight_dtype) + (sigma_t - sigma_s).view(-1, 1, 1, 1).to(weight_dtype) * d

    # target: same student, stop-gradient
    with torch.no_grad():
        target = edm_forward(
            x_t.float(), sigma_t, text_emb,
            student_dit.forward_without_cfg,
            edm_config
        )

    # loss
    if loss_type == "l2":
        loss = F.mse_loss(student_pred.float(), target.float())
    elif loss_type == "huber":
        loss = torch.sqrt(
            (student_pred.float() - target.float()) ** 2 + huber_c ** 2
        ) - huber_c
        loss = loss.mean()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    return loss
