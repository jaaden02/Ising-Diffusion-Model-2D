import os
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def binarize_sign(raw_x0):
    return np.sign(raw_x0).astype(np.float32)

def binarize_margin(raw_x0, delta=0.3):
    spins = np.sign(raw_x0).astype(np.float32)
    uncertain = np.abs(raw_x0) < delta
    probs = 1.0 / (1.0 + np.exp(-raw_x0[uncertain] / (delta / 3.0)))
    spins[uncertain] = np.where(np.random.rand(uncertain.sum()) < probs, 1.0, -1.0)
    return spins

def compute_energy(configs):
    right = np.roll(configs, 1, axis=2)
    down = np.roll(configs, 1, axis=1)
    E_per_sample = -(configs * right + configs * down).sum(axis=(1, 2))
    N = configs.shape[1] * configs.shape[2]
    return E_per_sample / N

def compute_magnetization(configs):
    N = configs.shape[1] * configs.shape[2]
    return np.abs(configs.reshape(configs.shape[0], -1).sum(axis=1)) / N

def compute_binder_cumulant(configs):
    N = configs.shape[1] * configs.shape[2]
    m = configs.reshape(configs.shape[0], -1).sum(axis=1) / N
    m2 = np.mean(m**2)
    m4 = np.mean(m**4)
    if m2 == 0:
        return 0.0
    return 1.0 - m4 / (3.0 * (m2**2))

def compute_correlation_function(configs, max_r=None):
    n_samples, L, _ = configs.shape
    if max_r is None:
        max_r = L // 2

    power_spectrum = np.zeros((L, L), dtype=np.float64)
    for i in range(n_samples):
        ft = np.fft.fft2(configs[i].astype(np.float64))
        power_spectrum += np.abs(ft)**2
    power_spectrum /= n_samples

    corr_2d = np.real(np.fft.ifft2(power_spectrum)) / (L * L)

    r_values = np.arange(0, max_r + 1)
    G_r = np.zeros(max_r + 1)
    counts = np.zeros(max_r + 1)

    for dx in range(L):
        for dy in range(L):
            rx = min(dx, L - dx)
            ry = min(dy, L - dy)
            r = np.sqrt(rx**2 + ry**2)
            r_int = int(round(r))
            if r_int <= max_r:
                G_r[r_int] += corr_2d[dx, dy]
                counts[r_int] += 1

    mask = counts > 0
    G_r[mask] /= counts[mask]
    return r_values, G_r

def fit_correlation_length(r_values, G_r, r_min=2, r_max=None):
    if r_max is None:
        r_max = len(r_values) - 1

    mask = (r_values >= r_min) & (r_values <= r_max) & (G_r > 0)
    r_fit = r_values[mask].astype(np.float64)
    log_G = np.log(G_r[mask])

    if len(r_fit) < 3:
        return np.nan

    A = np.vstack([np.ones_like(r_fit), r_fit]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, log_G, rcond=None)
    
    xi = -1.0 / coeffs[1] if coeffs[1] < 0 else np.nan
    return xi

def compute_all_observables(configs, T):
    N = configs.shape[1] * configs.shape[2]
    energies = compute_energy(configs)
    magnetizations = compute_magnetization(configs)

    E_mean = energies.mean()
    M_mean = magnetizations.mean()
    C_v = N * energies.var() / (T**2) if T > 0 else 0.0
    chi = N * magnetizations.var() / T if T > 0 else 0.0
    U_4 = compute_binder_cumulant(configs)

    r_values, G_r = compute_correlation_function(configs)
    xi = fit_correlation_length(r_values, G_r)

    return {
        "E": E_mean,
        "M": M_mean,
        "C_v": C_v,
        "chi": chi,
        "U_4": U_4,
        "xi": xi,
        "G_r": G_r.tolist(),
        "r_values": r_values.tolist()
    }

def evaluate_samples(samples_path, mc_data_path, output_dir, binarization="sign"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading generated samples from {samples_path}...")
    samples = np.load(samples_path)
    raw_x0 = samples["raw_x0"]
    temperatures = samples["temperatures"]

    if raw_x0.ndim == 5:
        raw_x0 = raw_x0.squeeze(2)

    binarize_fn = binarize_margin if binarization == "margin" else binarize_sign
    
    print(f"Loading Monte Carlo ground truth from {mc_data_path}...")
    mc_data = np.load(mc_data_path)
    mc_configs = mc_data["configs"]
    mc_temps = mc_data["temperatures"]

    print("Computing Monte Carlo ground truth observables...")
    mc_obs = {}
    for idx, T in enumerate(tqdm(mc_temps, desc="MC Observables")):
        mc_obs[float(T)] = compute_all_observables(mc_configs[idx], float(T))

    print("Computing generated model observables...")
    gen_obs = {}
    for idx, T in enumerate(tqdm(temperatures, desc="Model Observables")):
        bin_configs = binarize_fn(raw_x0[idx])
        gen_obs[float(T)] = compute_all_observables(bin_configs, float(T))

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"Ising Model Observables Comparison (Binarization: {binarization})", fontsize=14)

    quantities = [
        ("E", "Energy per spin $E/N$"),
        ("M", "Magnetization $|M|/N$"),
        ("C_v", "Specific heat $C_v$"),
        ("chi", r"Susceptibility $\chi$"),
        ("U_4", "Binder cumulant $U_4$"),
        ("xi", r"Correlation length $\xi$"),
    ]

    for ax, (key, ylabel) in zip(axes.flat, quantities):
        mc_vals = [mc_obs[float(T)][key] for T in mc_temps]
        gen_vals = [gen_obs[float(T)][key] for T in temperatures]

        ax.plot(mc_temps, mc_vals, "o-", label="MC Ground Truth", color="teal", alpha=0.8)
        ax.plot(temperatures, gen_vals, "s--", label="Model Generated", color="tomato", alpha=0.8)
        ax.axvline(2.269, color="gray", linestyle=":", label="$T_c \approx 2.27$")
        ax.set_xlabel("Temperature $T$")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plot_path = output_dir / f"observables_comparison_{binarization}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved observables comparison plot to: {plot_path}")
    plt.close()

    json_path = output_dir / "evaluation_summary.json"
    summary_data = {
        "temperatures": temperatures.tolist(),
        "mc": {str(T): {k: v for k, v in obs.items() if k not in ("G_r", "r_values")} for T, obs in mc_obs.items()},
        "model": {str(T): {k: v for k, v in obs.items() if k not in ("G_r", "r_values")} for T, obs in gen_obs.items()}
    }
    with open(json_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved evaluation metrics JSON summary to: {json_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate generated Ising configurations against Monte Carlo data")
    parser.add_argument("--samples", type=str, required=True, help="Path to model-generated samples (.npz)")
    parser.add_argument("--mc-data", type=str, required=True, help="Path to Monte Carlo comparison data (.npz)")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory where evaluation plots/logs will be saved")
    parser.add_argument("--binarization", type=str, default="sign", choices=["sign", "margin"], help="Threshold binarization method")
    
    args = parser.parse_args()
    evaluate_samples(args.samples, args.mc_data, args.output_dir, args.binarization)

if __name__ == "__main__":
    main()
