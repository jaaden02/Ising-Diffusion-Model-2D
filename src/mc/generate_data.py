"""
Generate Monte Carlo training data for the diffusion model.

Uses the Swendsen–Wang cluster algorithm to produce thermalized,
decorrelated spin configurations across a temperature grid spanning
the ferromagnetic phase transition.

Usage:
    python -m src.mc.generate_data --n-samples 2000 --n-temps 51
    python -m src.mc.generate_data --n-samples 5000 --n-temps 101 --L 64
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.mc.ising_model import IsingMC_SW


def generate(n_samples, n_temps, T_min=1.0, T_max=4.0, L=64, n_therm=1000, output_dir="src/data"):
    temperatures = np.linspace(T_min, T_max, n_temps)
    configs = np.zeros((n_temps, n_samples, L, L), dtype=np.int8)

    print(f"Generating {n_temps} × {n_samples} configs on L={L} lattice")
    print(f"Temperature range: [{T_min}, {T_max}], T_c ≈ 2.269")

    for i, T in enumerate(tqdm(temperatures, desc="Swendsen–Wang sampling")):
        model = IsingMC_SW(length=L, temperature=T)
        model.thermalize(n_therm)
        configs[i] = model.sample_configs(n_sample=n_samples, n_decorr=1)

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"ising_data_{n_temps}T_{n_samples}configs.npz"

    np.savez_compressed(
        out_path,
        configs=configs,
        temperatures=temperatures,
        run_id=np.array(run_id),
    )

    metadata = {
        "run_id": run_id,
        "L": L,
        "n_samples": n_samples,
        "n_therm": n_therm,
        "n_decorr": 1,
        "algorithm": "Swendsen–Wang (Numba JIT)",
        "temperatures": temperatures.tolist(),
        "n_temps": n_temps,
    }
    meta_path = out_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved: {out_path} ({size_mb:.1f} MB)")
    print(f"Metadata: {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Ising MC training data via Swendsen–Wang")
    parser.add_argument("--n-samples", type=int, default=2000, help="Configs per temperature")
    parser.add_argument("--n-temps", type=int, default=51, help="Number of temperature points")
    parser.add_argument("--T-min", type=float, default=1.0)
    parser.add_argument("--T-max", type=float, default=4.0)
    parser.add_argument("--L", type=int, default=64, help="Lattice side length")
    parser.add_argument("--n-therm", type=int, default=1000, help="Thermalization sweeps")
    parser.add_argument("--output-dir", type=str, default="src/data")
    args = parser.parse_args()
    generate(args.n_samples, args.n_temps, args.T_min, args.T_max, args.L, args.n_therm, args.output_dir)


if __name__ == "__main__":
    main()
