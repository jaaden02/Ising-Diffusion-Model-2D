import torch
import numpy as np
from torch.utils.data import Dataset

def d4_augment(config):
    """
    Apply random rotation and horizontal flip to augment a 2D square lattice configuration.
    This corresponds to the D4 point group symmetry of the square lattice, under which 
    the Ising Hamiltonian on a square lattice is invariant.
    """
    k = torch.randint(0, 4, (1,)).item()
    config = torch.rot90(config, k, dims=[-2, -1])
    if torch.rand(1).item() < 0.5:
        config = torch.flip(config, dims=[-1])
    return config

def spin_inv_augment(config):
    """
    Apply global spin inversion (up/down flip) to augment a configuration.
    Under zero external magnetic field, the Ising Hamiltonian is invariant under global 
    reversal of all spins (Z2 symmetry): s_i -> -s_i.
    """
    if torch.rand(1).item() < 0.5:
        config = -config
    return config

class IsingDataset(Dataset):
    """
    PyTorch Dataset loading generated Monte Carlo Ising spin configurations and 
    associated physical temperatures.
    """
    def __init__(self, path, augment=True):
        data = np.load(path)
        raw = data["configs"]          # (n_T, n_samples, L, L)
        t_phys = data["temperatures"]  # (n_T,)
        
        self.n_T, self.n_samples = raw.shape[:2]
        self.L = raw.shape[2]
        self.augment = augment
        
        self.configs = torch.from_numpy(
            raw.reshape(-1, self.L, self.L).astype(np.float32)
        )
        self.t_phys = torch.from_numpy(
            np.repeat(t_phys, self.n_samples).astype(np.float32)
        )

    def __len__(self):
        return self.configs.shape[0]

    def __getitem__(self, idx):
        config = self.configs[idx]
        
        if self.augment:
            config = d4_augment(config)
            config = spin_inv_augment(config)
            
        return config.unsqueeze(0), self.t_phys[idx]

    def get_configs_at_T(self, t_idx):
        start = t_idx * self.n_samples
        return self.configs[start : start + self.n_samples]

    def get_temperature_grid(self):
        return self.t_phys[:: self.n_samples]
