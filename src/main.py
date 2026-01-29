# -*- coding: utf-8 -*-
"""
Monte Carlo: Relativistic pion decay survival fraction

Simulates pion decay times in the lab frame using an exponential distribution,
computes traveled distance, and estimates the survival fraction for:
(a) fixed kinetic energy
(b) kinetic energy sampled from a Gaussian (truncated at K>0)

Author: Benjamín Rodríguez Valdez
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Params:
    c: float = 3.0e8         # m/s
    m: float = 139.6         # MeV (use consistent units with K)
    tau: float = 2.6e-8      # s (proper lifetime)
    L: float = 20.0          # m (detector distance)
    N: int = 1_000_000       # trials


def compute_beta_gamma(K: np.ndarray, m: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (beta, gamma) from kinetic energy K and rest mass m (same energy units)."""
    E = K + m
    gamma = E / m
    # Numerical safety: gamma >= 1 always if K>=0, but clip to avoid tiny negatives
    inv_g2 = 1.0 / np.maximum(gamma, 1.0) ** 2
    beta = np.sqrt(np.clip(1.0 - inv_g2, 0.0, 1.0))
    return beta, gamma


def simulate_survival_fraction(
    rng: np.random.Generator,
    p: Params,
    K: np.ndarray,
) -> tuple[int, int, float]:
    """
    Simulate decay times and compute survivors reaching distance L.
    Returns (survivors, N_eff, fraction).
    """
    beta, gamma = compute_beta_gamma(K, p.m)
    v = beta * p.c
    tau_lab = gamma * p.tau

    # Exponential decay: t = -tau_lab * ln(r), r ~ U(0,1)
    r = rng.random(len(K))
    t_decay = -tau_lab * np.log(r)

    distance = v * t_decay
    survivors = int(np.sum(distance >= p.L))
    N_eff = int(len(K))
    frac = survivors / N_eff if N_eff > 0 else float("nan")
    return survivors, N_eff, frac


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Monte Carlo simulation of relativistic pion decay survival fraction."
    )
    ap.add_argument("--N", type=int, default=1_000_000, help="Number of Monte Carlo trials (default: 1e6).")
    ap.add_argument("--L", type=float, default=20.0, help="Distance to detector in meters (default: 20).")
    ap.add_argument("--K", type=float, default=200.0, help="Kinetic energy mean in MeV (default: 200).")
    ap.add_argument("--sigmaK", type=float, default=50.0, help="Std dev of kinetic energy in MeV (default: 50).")
    ap.add_argument("--seed", type=int, default=12345, help="Random seed for reproducibility.")
    args = ap.parse_args()

    p = Params(L=args.L, N=args.N)
    rng = np.random.default_rng(args.seed)

    # (a) Fixed K
    K_fixed = np.full(p.N, float(args.K), dtype=float)
    surv_a, N_a, frac_a = simulate_survival_fraction(rng, p, K_fixed)

    print("Case (a) Fixed kinetic energy")
    print(f"  N trials        : {N_a}")
    print(f"  Survivors (>=L) : {surv_a}")
    print(f"  Fraction        : {frac_a:.6f}")
    print()

    # (b) Gaussian K, truncated at K>0
    K_gauss = rng.normal(loc=float(args.K), scale=float(args.sigmaK), size=p.N)
    K_gauss = K_gauss[K_gauss > 0.0]
    surv_b, N_b, frac_b = simulate_survival_fraction(rng, p, K_gauss)

    print("Case (b) Gaussian kinetic energy (truncated at K>0)")
    print(f"  N effective     : {N_b}")
    print(f"  Survivors (>=L) : {surv_b}")
    print(f"  Fraction        : {frac_b:.6f}")


if __name__ == "__main__":
    main()
