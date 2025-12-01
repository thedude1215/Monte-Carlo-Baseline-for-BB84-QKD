#!/usr/bin/env python3
"""
Enhanced BB84 QKD Simulation with Improved Robustness and Validation.
Reproduces and extends BB84 analysis with comprehensive error handling.

Reference: "A Validated and Reproducible Monte Carlo Baseline for the BB84 Protocol"
Author: Arnav Kumar
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple, List, Dict, Optional
import random
from dataclasses import dataclass
import time
import sys
import os
import warnings
from scipy import stats

# Version and dependencies
__version__ = "2.1.0"

# Validate dependencies early
try:
    from scipy.stats import t
except ImportError:
    print("Error: scipy is required but not installed.")
    print("Install with: pip install scipy")
    sys.exit(1)

@dataclass
class QKDResults:
    """Enhanced data structure to store QKD simulation results with statistics."""
    noise_level: float
    qber_mean: float
    qber_std: float
    qber_ci: float
    secure_key_rate: float
    skr_std: float
    skr_ci: float
    theory_qber: float
    sift_efficiency: float

class BB84SimulatorEnhanced:
    """
    Enhanced BB84 simulator with improved validation and error handling.
    Implements the core quantum transmission and measurement logic.
    """
    
    def __init__(self, key_length: int = 10000):
        if not isinstance(key_length, int) or key_length <= 0:
            raise ValueError(f"key_length must be a positive integer, got {key_length}")
        
        self.key_length = key_length
        # State definitions for reference (Z-basis: 0,1; X-basis: +,-)
        self.states = {
            (0, 0): '|0>', (1, 0): '|1>',
            (0, 1): '|+>', (1, 1): '|->'
        }
    
    def calculate_entropy(self, qber: float) -> float:
        """
        Calculate binary entropy H2(x) with improved edge case handling.
        H2(x) = -x*log2(x) - (1-x)*log2(1-x)
        """
        # Clip to avoid log(0) errors
        x = np.clip(qber, 1e-10, 1 - 1e-10)
        return -x * np.log2(x) - (1 - x) * np.log2(1 - x)

    def run_trial(self, noise_prob: float) -> Tuple[float, float, float]:
        """
        Run a single BB84 trial with vectorized operations for performance.
        
        Args:
            noise_prob: Depolarizing channel noise probability [0, 1]
            
        Returns:
            (qber, secure_rate, sift_efficiency)
        """
        if not (0 <= noise_prob <= 1):
            raise ValueError(f"noise_prob must be in [0,1], got {noise_prob}")
        
        # 1. Alice Preparation (Vectorized)
        alice_bits = np.random.randint(0, 2, self.key_length)
        alice_bases = np.random.randint(0, 2, self.key_length)
        
        # 2. Channel Transmission (Depolarizing Noise)
        # Generate noise mask: True where noise occurs
        noise_mask = np.random.random(self.key_length) < noise_prob
        
        # 3. Bob Measurement
        bob_bases = np.random.randint(0, 2, self.key_length)
        bob_bits = alice_bits.copy()
        
        # Apply noise: For affected qubits, replace with random outcome
        # This models the "completely mixed state" I/2 resulting from depolarization
        num_noise_events = np.sum(noise_mask)
        if num_noise_events > 0:
            noise_indices = np.where(noise_mask)[0]
            bob_bits[noise_indices] = np.random.randint(0, 2, num_noise_events)
        
        # 4. Sifting (Basis Reconciliation)
        matching_bases = (alice_bases == bob_bases)
        sifted_alice = alice_bits[matching_bases]
        sifted_bob = bob_bits[matching_bases]
        
        sifted_len = len(sifted_alice)
        if sifted_len == 0:
            return 0.0, 0.0, 0.0
            
        # 5. Calculate Metrics
        errors = np.sum(sifted_alice != sifted_bob)
        qber = errors / sifted_len
        sift_eff = sifted_len / self.key_length
        
        # Secure Key Rate (Shor-Preskill Bound)
        # R = 1 - 2*H2(Q) (assuming f_ec = 1.0)
        h2 = self.calculate_entropy(qber)
        secure_rate = max(0.0, 1.0 - 2.0 * h2)
        
        return qber, secure_rate, sift_eff

def run_experiment_sweep(start: float, stop: float, step: float, 
                        qubits: int, trials: int, seed: int) -> pd.DataFrame:
    """
    Executes the full experimental sweep with statistical aggregation.
    """
    print(f"\nStarting Enhanced BB84 Analysis (v{__version__})")
    print("=" * 65)
    print(f"Parameters: Qubits={qubits:,}, Trials={trials}, Seed={seed}")
    print(f"Sweep Range: {start:.2f} -> {stop:.2f} (step {step:.2f})")
    print("-" * 65)
    print(f"{'Noise':<8} | {'QBER (mean)':<12} | {'Secure Rate':<12} | {'Theory Q'}")
    print("-" * 65)
    
    # Set seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    
    # Generate noise levels safely
    num_steps = int(round((stop - start) / step)) + 1
    noise_levels = np.linspace(start, stop, num_steps)
    
    simulator = BB84SimulatorEnhanced(key_length=qubits)
    results_data = []
    
    start_time = time.time()
    
    for p in noise_levels:
        trial_qbers = []
        trial_rates = []
        trial_sifts = []
        
        for _ in range(trials):
            q, r, s = simulator.run_trial(p)
            trial_qbers.append(q)
            trial_rates.append(r)
            trial_sifts.append(s)
            
        # Calculate Statistics
        mean_qber = np.mean(trial_qbers)
        mean_rate = np.mean(trial_rates)
        mean_sift = np.mean(trial_sifts)
        
        # Standard Deviation & Confidence Intervals
        std_qber = np.std(trial_qbers, ddof=1) if trials > 1 else 0.0
        std_rate = np.std(trial_rates, ddof=1) if trials > 1 else 0.0
        
        if trials > 1:
            # 95% CI using t-distribution
            t_crit = stats.t.ppf(0.975, trials - 1)
            ci_qber = t_crit * (std_qber / np.sqrt(trials))
            ci_rate = t_crit * (std_rate / np.sqrt(trials))
        else:
            ci_qber = 0.0
            ci_rate = 0.0
            
        theory_q = p / 2.0
        
        # Log to console
        print(f"{p:<8.2f} | {mean_qber:<12.4f} | {mean_rate:<12.4f} | {theory_q:.4f}")
        
        results_data.append({
            "p_depolarizing": p,
            "qber_mean": mean_qber,
            "qber_std": std_qber,
            "qber_ci": ci_qber,
            "secure_rate_mean": mean_rate,
            "secure_rate_std": std_rate,
            "secure_rate_ci": ci_rate,
            "sift_efficiency": mean_sift,
            "theory_qber": theory_q
        })
        
    total_time = time.time() - start_time
    print("-" * 65)
    print(f"Experiment completed in {total_time:.2f} seconds.")
    
    return pd.DataFrame(results_data)

def generate_plots(df: pd.DataFrame, output_dir: str):
    """
    Generates publication-quality plots from the results DataFrame.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. QBER Validation Plot
    plt.figure(figsize=(10, 6), dpi=100)
    plt.errorbar(df['p_depolarizing'], df['qber_mean'], yerr=df['qber_ci'], 
                 fmt='o', capsize=3, label='Simulation (95% CI)', color='#1f77b4')
    plt.plot(df['p_depolarizing'], df['theory_qber'], '--', 
             label='Theory (Q = p/2)', color='orange', linewidth=2)
    plt.axhline(0.11, color='red', linestyle=':', label='Security Threshold (~11%)')
    
    plt.title("Validation of QBER vs. Depolarizing Noise")
    plt.xlabel("Depolarizing Probability (p)")
    plt.ylabel("Quantum Bit Error Rate (QBER)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'qber_plot.png'))
    plt.close()
    
    # 2. Secure Key Rate Plot
    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(df['p_depolarizing'], df['secure_rate_mean'], 'o-', 
             color='green', label='Simulated Rate', linewidth=2)
    
    plt.title("Secure Key Rate (Shor-Preskill Bound)")
    plt.xlabel("Depolarizing Probability (p)")
    plt.ylabel("Secure Bits per Sifted Bit")
    plt.axvline(0.22, color='red', linestyle='--', label='Critical Noise Limit (p~0.22)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'secure_key_rate.png'))
    plt.close()
    
    print(f"Plots saved to directory: {output_dir}/")

def main():
    """
    Main entry point. Parses CLI arguments and orchestrates the simulation.
    """
    parser = argparse.ArgumentParser(
        description="BB84 QKD Monte Carlo Simulator - Validated Baseline"
    )
    
    # Simulation Parameters
    parser.add_argument("--qubits", type=int, default=10000, 
                       help="Number of qubits per trial (default: 10000)")
    parser.add_argument("--trials", type=int, default=10, 
                       help="Number of independent trials per noise point (default: 10)")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed for reproducibility (default: 42)")
    
    # Sweep Range Parameters
    parser.add_argument("--start", type=float, default=0.00, 
                       help="Start noise probability (default: 0.0)")
    parser.add_argument("--stop", type=float, default=0.26, 
                       help="End noise probability (default: 0.26)")
    parser.add_argument("--step", type=float, default=0.02, 
                       help="Step size for noise sweep (default: 0.02)")
    
    # Output Control
    parser.add_argument("--output", type=str, default="data/results.csv", 
                       help="Path to save results CSV (default: data/results.csv)")
    parser.add_argument("--figures", action="store_true", 
                       help="Generate validation plots")
    
    args = parser.parse_args()
    
    # Run the core logic
    try:
        results_df = run_experiment_sweep(
            args.start, args.stop, args.step, 
            args.qubits, args.trials, args.seed
        )
        
        # Save Data
        if args.output:
            out_dir = os.path.dirname(args.output)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            results_df.to_csv(args.output, index=False)
            print(f"Results saved to: {args.output}")
            
        # Generate Plots
        if args.figures:
            # Determine figure directory relative to output or default to 'figures'
            if args.output:
                base_dir = os.path.dirname(args.output)
                # If output is in data/, put figures in figures/ (sibling dir)
                if os.path.basename(base_dir) == 'data':
                    fig_dir = os.path.join(os.path.dirname(base_dir), 'figures')
                else:
                    fig_dir = os.path.join(base_dir, 'figures')
            else:
                fig_dir = 'figures'
                
            generate_plots(results_df, fig_dir)
            
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()