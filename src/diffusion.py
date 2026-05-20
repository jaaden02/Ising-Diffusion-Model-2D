import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class Diffuser(nn.Module):
    """
    DDPM and DDIM noise scheduler and reverse diffusion sampler.
    """
    def __init__(self, diff_steps=1000, schedule="cosine", beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.diff_steps = diff_steps
        self.schedule_name = schedule
        
        betas = self.make_schedule(schedule, beta_start, beta_end)
        alphas = 1.0 - betas
        alpha_hats = torch.cumprod(alphas, dim=0)
        
        alpha_hats_prev = F.pad(alpha_hats[:-1], (1, 0), value=1.0)
        posterior_variance = betas * (1.0 - alpha_hats_prev) / (1.0 - alpha_hats)
        posterior_log_var = torch.log(torch.clamp(posterior_variance, min=1e-20))

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_hats", alpha_hats)
        self.register_buffer("alpha_hats_prev", alpha_hats_prev)
        self.register_buffer("sqrt_alpha_hats", torch.sqrt(alpha_hats))
        self.register_buffer("sqrt_one_minus_alpha_hats", torch.sqrt(1.0 - alpha_hats))
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_var", posterior_log_var)

    def make_schedule(self, schedule, beta_start, beta_end):
        if schedule == "linear":
            return torch.linspace(beta_start, beta_end, self.diff_steps)
        elif schedule == "cosine":
            s = 0.008
            steps = torch.arange(self.diff_steps + 1, dtype=torch.float32)
            alpha_bar = torch.cos(((steps / self.diff_steps) + s) / (1 + s) * torch.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            return torch.clamp(1 - alpha_bar[1:] / alpha_bar[:-1], max=0.999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

    def diffuse(self, x0, t):
        eps = torch.randn_like(x0)
        sqrt_alpha_t = self.sqrt_alpha_hats[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alpha_hats[t].view(-1, 1, 1, 1)
        x_t = sqrt_alpha_t * x0 + sqrt_one_minus_alpha_t * eps
        return x_t, eps

    def predicted_log_var(self, v, t):
        log_beta = torch.log(torch.clamp(self.betas[t], min=1e-20)).view(-1, 1, 1, 1)
        post_log_var = self.posterior_log_var[t].view(-1, 1, 1, 1)
        v = torch.sigmoid(v)
        return v * log_beta + (1 - v) * post_log_var

    def vlb_loss(self, x0, x_t, t, eps_pred, v_pred):
        alpha_hat_t = self.alpha_hats[t].view(-1, 1, 1, 1)
        alpha_hat_prev = self.alpha_hats_prev[t].view(-1, 1, 1, 1)
        beta_t = self.betas[t].view(-1, 1, 1, 1)
        alpha_t = self.alphas[t].view(-1, 1, 1, 1)

        posterior_mean = (
            torch.sqrt(alpha_hat_prev) * beta_t / (1 - alpha_hat_t) * x0
            + torch.sqrt(alpha_t) * (1 - alpha_hat_prev) / (1 - alpha_hat_t) * x_t
        )

        model_mean = (1 / torch.sqrt(alpha_t)) * (
            x_t - beta_t / torch.sqrt(1 - alpha_hat_t) * eps_pred
        )

        model_log_var = self.predicted_log_var(v_pred, t)
        post_log_var = self.posterior_log_var[t].view(-1, 1, 1, 1)

        kl = 0.5 * (
            model_log_var - post_log_var
            + torch.exp(post_log_var - model_log_var)
            + (posterior_mean - model_mean) ** 2 * torch.exp(-model_log_var)
            - 1.0
        )
        return kl.mean()

    @torch.no_grad()
    def sample(self, model, n_samples, T_phys, use_predicted_var=False):
        device = next(model.parameters()).device
        model.eval()

        if isinstance(T_phys, (int, float)):
            T_phys = torch.full((n_samples,), T_phys, device=device)
        else:
            T_phys = T_phys.to(device)

        x = torch.randn(n_samples, 1, 64, 64, device=device)

        for t_step in tqdm(reversed(range(self.diff_steps)), desc="Sampling", leave=False, total=self.diff_steps):
            t = torch.full((n_samples,), t_step, dtype=torch.long, device=device)

            model_output = model(x, t.float(), T_phys)
            pred_noise, v_pred = model_output.chunk(2, dim=1)

            alpha = self.alphas[t_step]
            alpha_hat = self.alpha_hats[t_step]
            beta = self.betas[t_step]

            if use_predicted_var:
                log_var_pred = self.predicted_log_var(v_pred, t)
                noise_scale = torch.exp(0.5 * log_var_pred)
            else:
                noise_scale = torch.sqrt(beta)

            noise = torch.randn_like(x) if t_step > 0 else torch.zeros_like(x)
            x = (1.0 / torch.sqrt(alpha)) * (
                x - (1.0 - alpha) / torch.sqrt(1.0 - alpha_hat) * pred_noise
            ) + noise_scale * noise

        model.train()
        return x.clamp(-1.0, 1.0)

    @torch.no_grad()
    def sample_respaced(self, model, n_samples, T_phys, n_steps=100, use_predicted_var=False):
        device = next(model.parameters()).device
        model.eval()

        if isinstance(T_phys, (int, float)):
            T_phys = torch.full((n_samples,), T_phys, device=device)
        else:
            T_phys = T_phys.to(device)

        respaced_timesteps = torch.linspace(0, self.diff_steps - 1, n_steps + 1).long()
        x = torch.randn(n_samples, 1, 64, 64, device=device)

        for i in tqdm(reversed(range(n_steps)), desc=f"Respaced Sampling ({n_steps} steps)", leave=False, total=n_steps):
            t_current = respaced_timesteps[i + 1].item()
            t_prev = respaced_timesteps[i].item()

            t_tensor = torch.full((n_samples,), t_current, dtype=torch.long, device=device)

            model_output = model(x, t_tensor.float(), T_phys)
            pred_noise, v_pred = model_output.chunk(2, dim=1)

            alpha_hat_t = self.alpha_hats[t_current]
            alpha_hat_prev = self.alpha_hats[t_prev] if t_prev > 0 else torch.tensor(1.0, device=device)
            beta_respaced = 1.0 - alpha_hat_t / alpha_hat_prev

            if t_prev > 0:
                posterior_var = beta_respaced * (1.0 - alpha_hat_prev) / (1.0 - alpha_hat_t)
            else:
                posterior_var = torch.tensor(0.0, device=device)

            x0_pred = (x - torch.sqrt(1.0 - alpha_hat_t) * pred_noise) / torch.sqrt(alpha_hat_t)
            x0_pred = x0_pred.clamp(-1.0, 1.0)

            coeff_x0 = torch.sqrt(alpha_hat_prev) * beta_respaced / (1.0 - alpha_hat_t)
            coeff_xt = torch.sqrt(1.0 - beta_respaced) * (1.0 - alpha_hat_prev) / (1.0 - alpha_hat_t)
            mean = coeff_x0 * x0_pred + coeff_xt * x

            if use_predicted_var and t_prev > 0:
                log_var_pred = self.predicted_log_var(v_pred, t_tensor)
                noise_scale = torch.exp(0.5 * log_var_pred)
            else:
                noise_scale = torch.sqrt(posterior_var) if t_prev > 0 else 0.0

            noise = torch.randn_like(x) if t_prev > 0 else torch.zeros_like(x)
            x = mean + noise_scale * noise

        model.train()
        return x.clamp(-1.0, 1.0)

    @torch.no_grad()
    def sample_ddim(self, model, n_samples, T_phys, n_steps=50, eta=0.0):
        device = next(model.parameters()).device
        model.eval()

        if isinstance(T_phys, (int, float)):
            T_phys = torch.full((n_samples,), T_phys, device=device)
        else:
            T_phys = T_phys.to(device)

        tau = torch.linspace(0, self.diff_steps - 1, n_steps + 1).long()
        x = torch.randn(n_samples, 1, 64, 64, device=device)

        for i in tqdm(reversed(range(n_steps)), desc=f"DDIM ({n_steps} steps)", leave=False, total=n_steps):
            t_cur = tau[i + 1].item()
            t_prev = tau[i].item()

            t_tensor = torch.full((n_samples,), t_cur, dtype=torch.long, device=device)

            model_output = model(x, t_tensor.float(), T_phys)
            eps_pred = model_output.chunk(2, dim=1)[0]

            alpha_hat_t = self.alpha_hats[t_cur]
            alpha_hat_prev = self.alpha_hats[t_prev] if t_prev > 0 else torch.tensor(1.0, device=device)

            x0_pred = (x - torch.sqrt(1.0 - alpha_hat_t) * eps_pred) / torch.sqrt(alpha_hat_t)
            x0_pred = x0_pred.clamp(-1.0, 1.0)

            dir_xt = torch.sqrt(1.0 - alpha_hat_prev - eta**2 * self._ddim_sigma(alpha_hat_t, alpha_hat_prev)**2) * eps_pred

            if eta > 0 and t_prev > 0:
                sigma = eta * self._ddim_sigma(alpha_hat_t, alpha_hat_prev)
                noise = torch.randn_like(x)
            else:
                sigma = 0.0
                noise = 0.0

            x = torch.sqrt(alpha_hat_prev) * x0_pred + dir_xt + sigma * noise

        model.train()
        return x.clamp(-1.0, 1.0)

    @staticmethod
    def _ddim_sigma(alpha_hat_t, alpha_hat_prev):
        return torch.sqrt((1.0 - alpha_hat_prev) / (1.0 - alpha_hat_t) * (1.0 - alpha_hat_t / alpha_hat_prev))
