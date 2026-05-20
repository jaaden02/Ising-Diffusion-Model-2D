"""
Monte Carlo simulation of the 2D Ising Model on a square lattice
with periodic boundary conditions.

Implements two update algorithms:
  - Single-spin Metropolis–Hastings (IsingMC_Metropolis)
  - Swendsen–Wang cluster algorithm  (IsingMC_SW)

The Swendsen–Wang algorithm uses Fortuin–Kasteleyn bond percolation
and Hoshen–Kopelman cluster labeling with union-find, dramatically
reducing critical slowing down near T_c ≈ 2.269.

All inner loops are JIT-compiled via Numba for near-C performance.
"""
import numpy as np
from numba import njit


# ---------------------------------------------------------------------------
#  Base class
# ---------------------------------------------------------------------------

class IsingMC:
    """Base Ising Monte Carlo sampler on an L×L square lattice."""

    def __init__(self, length, temperature=0.0):
        self.spins = np.ones((length, length), dtype=int)
        self.L = length
        self.J = 1.0
        self.T = temperature
        self.M = length * length

    def set_temperature(self, temperature):
        self.T = temperature
        self.update_probabilities()

    def reset_spins(self):
        self.spins.fill(1)
        self.M = self.L * self.L

    def thermalize(self, n_therm):
        """Discard n_therm sweeps to reach thermal equilibrium."""
        for _ in range(n_therm):
            self.step()

    def step(self):
        raise NotImplementedError

    def sample(self, n_sample, n_decorr=1):
        M_abs = np.zeros(n_sample)
        M_raw = np.zeros(n_sample)
        E_samples = np.zeros(n_sample)
        for i in range(n_sample):
            for _ in range(n_decorr):
                self.step()
            mag = np.sum(self.spins)
            M_raw[i] = mag
            M_abs[i] = np.abs(mag)
            E_samples[i] = self.energy()
        return M_abs, M_raw, E_samples

    def sample_configs(self, n_sample, n_decorr=1):
        """Generate n_sample decorrelated spin configurations."""
        configs = np.zeros((n_sample, self.L, self.L), dtype=np.int8)
        for i in range(n_sample):
            for _ in range(n_decorr):
                self.step()
            configs[i] = self.spins
        return configs

    def calibrate_decorr(self, n_cal, tau_factor=2):
        """Estimate decorrelation time via binning analysis."""
        n_cal = 1 << int(np.log2(n_cal))

        M_series = np.zeros(n_cal)
        for i in range(n_cal):
            self.step()
            M_series[i] = np.sum(self.spins)

        max_level = int(np.log2(n_cal / 30))
        deltas = []
        data = M_series.copy()

        for l in range(max_level + 1):
            M_l = len(data)
            delta = np.sqrt(np.sum((data - np.mean(data)) ** 2) / (M_l * (M_l - 1)))
            deltas.append(delta)
            if l < max_level:
                data = (data[::2] + data[1::2]) / 2

        deltas = np.array(deltas)
        delta_0 = deltas[0]
        delta_plateau = _find_plateau_delta(deltas)

        tau = max(0.5, 0.5 * ((delta_plateau / delta_0) ** 2 - 1))
        n_decorr = max(1, int(tau_factor * tau))
        return n_decorr, tau

    def energy(self):
        return _total_energy(self.spins, self.J, self.L)

    def magnetization(self):
        return np.sum(self.spins)


def _find_plateau_delta(deltas, threshold=0.05):
    for i in range(len(deltas) - 1):
        ratio = deltas[i + 1] / deltas[i]
        if abs(ratio - 1) < threshold:
            return np.mean(deltas[i:])
    return deltas[-1]


# ---------------------------------------------------------------------------
#  Metropolis–Hastings
# ---------------------------------------------------------------------------

class IsingMC_Metropolis(IsingMC):
    """Single-spin Metropolis sampler with precomputed Boltzmann table."""

    def __init__(self, length, temperature=0.0):
        super().__init__(length, temperature)
        self._beta = None
        self._exp_table = None

    def step(self):
        beta = 1.0 / self.T if self.T != 0 else np.inf
        self.sweep(beta, n=1)

    def sweep(self, beta, n=1):
        if not hasattr(self, "_exp_table") or self._beta != beta:
            self._exp_table = _boltzmann_table(beta, self.J)
            self._beta = beta
        _sweep(self.spins, self._exp_table, self.L, n)


# ---------------------------------------------------------------------------
#  Swendsen–Wang cluster algorithm
# ---------------------------------------------------------------------------

class IsingMC_SW(IsingMC):
    """
    Swendsen–Wang cluster algorithm.
    
    At each step:
      1. Place Fortuin–Kasteleyn bonds between aligned neighbors
         with probability p = 1 − exp(−2J/T).
      2. Identify connected clusters via Hoshen–Kopelman labeling
         (union-find with path compression).
      3. Flip each cluster independently with probability 1/2.
    
    This eliminates critical slowing down: the integrated
    autocorrelation time τ remains O(1) even at T_c.
    """

    def __init__(self, length, temperature=0.0):
        super().__init__(length, temperature)
        self.sw_prob = None
        self.h_bonds = np.zeros((length, length), dtype=np.int32)
        self.v_bonds = np.zeros((length, length), dtype=np.int32)
        self.cluster = np.zeros((length, length), dtype=np.int32)
        self.parent = np.zeros(length * length, dtype=np.int32)
        self.update_probabilities()

    def update_probabilities(self):
        self.sw_prob = (1 - np.exp(-2 * self.J / self.T)) if self.T != 0 else 1.0

    def calibrate_decorr(self, n_cal):
        return 1, 0.5

    def step(self):
        self.h_bonds.fill(0)
        self.v_bonds.fill(0)
        _sw_bond_sweep(self.spins, self.L, self.sw_prob, self.h_bonds, self.v_bonds)
        _sw_hoshen_kopelman(
            self.h_bonds, self.v_bonds, self.cluster, self.parent, self.L
        )
        _sw_flip(self.spins, self.cluster, self.parent, self.L)
        self.M = np.sum(self.spins)


# ---------------------------------------------------------------------------
#  Numba-compiled kernels
# ---------------------------------------------------------------------------

@njit(cache=True)
def _total_energy(x, J, L):
    energy = 0.0
    for i in range(L):
        for j in range(L):
            energy += -J * x[i, j] * (x[(i + 1) % L, j] + x[i, (j + 1) % L])
    return energy

@njit(cache=True)
def _boltzmann_table(beta, J):
    table = np.empty(5)
    for k in range(5):
        dE = (-8 + 4 * k) * J
        table[k] = np.exp(-beta * dE) if dE > 0 else 1.0
    return table

@njit(cache=True)
def _move(x, exp_table, L):
    i = np.random.randint(0, L)
    j = np.random.randint(0, L)
    nn = x[(i - 1) % L, j] + x[(i + 1) % L, j] + x[i, (j - 1) % L] + x[i, (j + 1) % L]
    dE_idx = (x[i, j] * nn + 4) // 2
    if np.random.rand() < exp_table[dE_idx]:
        x[i, j] = -x[i, j]

@njit(cache=True)
def _sweep(x, exp_table, L, n_sweeps=1):
    for _ in range(n_sweeps * L * L):
        _move(x, exp_table, L)

@njit(cache=True)
def _sw_find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

@njit(cache=True)
def _sw_union(parent, a, b):
    ra = _sw_find(parent, a)
    rb = _sw_find(parent, b)
    if ra != rb:
        parent[rb] = ra
    return ra

@njit(cache=True)
def _sw_bond_sweep(spins, L, prob, h_bonds, v_bonds):
    for i in range(L):
        for j in range(L):
            if spins[i, j] == spins[(i + 1) % L, j] and np.random.rand() < prob:
                v_bonds[i, j] = 1
            if spins[i, j] == spins[i, (j + 1) % L] and np.random.rand() < prob:
                h_bonds[i, j] = 1

@njit(cache=True)
def _sw_hoshen_kopelman(h_bonds, v_bonds, cluster, parent, L):
    next_label = 0
    for i in range(L):
        for j in range(L):
            has_top = (i > 0) and v_bonds[i - 1, j] == 1
            has_left = (j > 0) and h_bonds[i, j - 1] == 1

            if not has_top and not has_left:
                cluster[i, j] = next_label
                parent[next_label] = next_label
                next_label += 1
            elif has_top and not has_left:
                cluster[i, j] = _sw_find(parent, cluster[i - 1, j])
            elif not has_top and has_left:
                cluster[i, j] = _sw_find(parent, cluster[i, j - 1])
            else:
                cluster[i, j] = _sw_union(parent, cluster[i - 1, j], cluster[i, j - 1])

    # periodic boundary stitching
    for j in range(L):
        if v_bonds[L - 1, j] == 1:
            _sw_union(parent, cluster[L - 1, j], cluster[0, j])
    for i in range(L):
        if h_bonds[i, L - 1] == 1:
            _sw_union(parent, cluster[i, L - 1], cluster[i, 0])

    # flatten labels
    for i in range(L):
        for j in range(L):
            cluster[i, j] = _sw_find(parent, cluster[i, j])

@njit(cache=True)
def _sw_flip(spins, cluster, parent, L):
    flip = parent.copy()
    flip[:] = -1
    for i in range(L):
        for j in range(L):
            root = cluster[i, j]
            if flip[root] == -1:
                flip[root] = 1 if np.random.rand() < 0.5 else 0
            if flip[root] == 1:
                spins[i, j] *= -1
