import os
import json
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from src.dataset import IsingDataset
from src.model import UNet_S
from src.diffusion import Diffuser

def load_model(checkpoint_path, device, use_ema=True):
    """Load model state dict from checkpoint, prioritizing EMA weights if available."""
    model = UNet_S().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        if use_ema and ckpt.get("ema") is not None:
            print("Applying shadow EMA weights from checkpoint.")
            state_dict = {}
            for name, param in model.named_parameters():
                if name in ckpt["ema"]:
                    state_dict[name] = ckpt["ema"][name]
                else:
                    state_dict[name] = ckpt["model"][name]
            model.load_state_dict(state_dict)
        else:
            print("Applying regular trained weights from checkpoint.")
            model.load_state_dict(ckpt["model"])
    else:
        print("Applying direct state dict weights from checkpoint.")
        model.load_state_dict(ckpt)

    model.eval()
    return model

def sample_batched(model, diffuser, temperatures, n_samples_per_T, device,
                   sampler="ddpm", n_steps=1000, eta=0.0, max_batch=256,
                   use_predicted_var=False):
    """
    Generate raw continuous samples across a grid of temperatures in chunks.
    """
    n_temps = len(temperatures)
    total_samples = n_temps * n_samples_per_T

    T_phys_full = torch.tensor(
        np.repeat(temperatures, n_samples_per_T), dtype=torch.float32, device=device
    )

    all_outputs = []
    n_chunks = (total_samples + max_batch - 1) // max_batch

    with torch.no_grad():
        for chunk_idx in tqdm(range(n_chunks), desc=f"Denoising chunks ({sampler})"):
            start = chunk_idx * max_batch
            end = min(start + max_batch, total_samples)
            T_chunk = T_phys_full[start:end]
            chunk_size = end - start

            if sampler == "ddpm":
                x = diffuser.sample(
                    model, n_samples=chunk_size, T_phys=T_chunk, use_predicted_var=use_predicted_var
                )
            elif sampler == "ddim":
                x = diffuser.sample_ddim(
                    model, n_samples=chunk_size, T_phys=T_chunk, n_steps=n_steps, eta=eta
                )
            elif sampler == "respaced":
                x = diffuser.sample_respaced(
                    model, n_samples=chunk_size, T_phys=T_chunk, n_steps=n_steps, use_predicted_var=use_predicted_var
                )
            else:
                raise ValueError(f"Unknown sampler strategy: {sampler}")

            all_outputs.append(x.cpu())

    raw_x0 = torch.cat(all_outputs, dim=0)  # Shape: (total_samples, 1, L, L)
    L = raw_x0.shape[-1]
    
    return raw_x0.view(n_temps, n_samples_per_T, 1, L, L)

def main():
    parser = argparse.ArgumentParser(description="Generate raw continuous spin configurations from a trained checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint model file (.pt)")
    parser.add_argument("--data", type=str, default=None, help="Path to MC dataset to extract the temperature grid")
    parser.add_argument("--temperatures", type=float, nargs="+", default=None, help="Explicit temperatures list to sample")
    parser.add_argument("--n-samples", type=int, default=100, help="Number of samples to generate per temperature")
    parser.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim", "respaced"], help="Sampling sampler method")
    parser.add_argument("--n-steps", type=int, default=1000, help="Number of denoising steps (for ddim/respaced)")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM noise injection scale (0.0=deterministic, 1.0=stochastic)")
    parser.add_argument("--max-batch", type=int, default=256, help="Maximum batch size chunk for denoising passes")
    parser.add_argument("--output-dir", type=str, default="samples", help="Directory where generated samples will be saved")
    parser.add_argument("--use-ema", action="store_true", default=True, help="Load shadow EMA weights if present in checkpoint")
    parser.add_argument("--use-predicted-var", action="store_true", help="Use model's learned VLB variance during sampling")
    
    args = parser.parse_args()

    if args.temperatures is not None:
        temperatures = np.array(args.temperatures, dtype=np.float32)
    elif args.data is not None:
        mc_data = np.load(args.data)
        temperatures = mc_data["temperatures"].astype(np.float32)
        print(f"Loaded temperature grid of {len(temperatures)} temperatures from {args.data}")
    else:
        parser.error("Please provide either --data (for grid) or --temperatures (list)")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Denoising running on device: {device}")

    model = load_model(args.checkpoint, device, use_ema=args.use_ema)
    diffuser = Diffuser().to(device)

    print(f"Sampling {args.n_samples} configurations at {len(temperatures)} temperatures...")
    raw_x0 = sample_batched(
        model, diffuser, temperatures, args.n_samples, device,
        sampler=args.sampler, n_steps=args.n_steps, eta=args.eta,
        max_batch=args.max_batch, use_predicted_var=args.use_predicted_var
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"generated_samples_{timestamp}.npz"

    np.savez_compressed(
        out_path,
        raw_x0=raw_x0.numpy(),
        temperatures=temperatures,
        sampler=args.sampler,
        n_steps=args.n_steps,
        checkpoint=str(Path(args.checkpoint).resolve())
    )
    
    print(f"Successfully generated and saved samples to: {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")

if __name__ == "__main__":
    main()
