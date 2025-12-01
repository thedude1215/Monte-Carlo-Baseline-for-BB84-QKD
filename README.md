# A Validated and Reproducible Monte Carlo Baseline for the BB84 Protocol

![License](https://img.shields.io/badge/license-Open%20Source-blue.svg)
![Python](https://img.shields.io/badge/python-3.x-blue.svg)
![Status](https://img.shields.io/badge/status-Validated-brightgreen.svg)

**Author:** Arnav Kumar  
[cite_start]**Date:** August 26, 2025 [cite: 3]

## 📌 Overview

[cite_start]This repository contains the source code and documentation for the study **"A Validated and Reproducible Monte Carlo Baseline for the BB84 Protocol Under Depolarizing Channel Noise."** [cite: 1]

This project establishes a statistically rigorous computational benchmark for the BB84 Quantum Key Distribution (QKD) protocol. [cite_start]By implementing a transparent, parameterized simulator, we quantify the performance of BB84 under symmetric depolarizing noise, providing a "computational null hypothesis" against which practical QKD systems can be measured. [cite: 5, 11]



## ✨ Key Features

* [cite_start]**Rigorous Statistical Analysis:** Results are derived from large-scale trials ($10^4$ signals) with 10 independent repetitions, including 95% confidence intervals. [cite: 7]
* [cite_start]**Full Noise Sweep:** Simulates depolarizing probabilities from $p=0\%$ to $p=26\%$. [cite: 6]
* [cite_start]**Theoretical Validation:** Validates the $Q \approx p/2$ relationship (max deviation 0.27%) and the security threshold collapse at $\approx 11\%$ QBER. [cite: 8]
* [cite_start]**Standardized Metrics:** Outputs Quantum Bit Error Rate (QBER) and Shor-Preskill secure key rates ($R_{QKD}$). [cite: 7]

## ⚙️ Methodology & Assumptions

[cite_start]The simulation operates under **idealized assumptions** to establish an upper bound on performance: [cite: 45]
1.  **Perfect Single-Photon Sources** (No multi-photon emissions).
2.  **Ideal Detectors** (Unit efficiency, zero dark counts).
3.  **Perfect Error Correction** ($f_{EC} = 1.0$).
4.  **Asymptotic Key Analysis** (No finite-size effects).

### Channel Model
[cite_start]The simulation utilizes a qubit depolarizing channel $\mathcal{E}(\rho)$ defined by the Kraus operator: [cite: 50, 51]

$$\mathcal{E}(\rho) = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)$$

[cite_start]For BB84, this reduces to an effective binary symmetric channel with bit-flip probability $p/2$. [cite: 57]

## 🚀 Usage

[cite_start]To reproduce the exact results presented in the paper, use the following command: [cite: 226, 227]

```bash
python enhanced_bb84_simulator.py --qubits 10000 --trials 10 \
 --start 0.00 --stop 0.26 --step 0.02 --seed 42 \
 --output results_enhanced.csv --figures --validation


## 📂 Repository Structure

The project is organized as follows:

```plaintext
BB84-Monte-Carlo-Benchmark/
│
├── README.md                       <-- Project documentation and results summary
│
├── manuscript/
│   └── BB84_Monte_Carlo_Paper.pdf  <-- Full research paper describing the methodology
│
├── src/                            <-- Source Code
[cite_start]│   ├── enhanced_bb84_simulator.py  <-- The main simulation engine [cite: 227]
│   └── requirements.txt            <-- Dependencies (numpy, matplotlib, pandas)
│
├── data/                           <-- Simulation Output
[cite_start]│   └── results_enhanced.csv        <-- Dataset with raw simulation results [cite: 227]
│
└── figures/                        <-- Visualization
    ├── qber_plot.png               <-- Figure 1: QBER vs. Depolarizing Probability
    └── throughput_plot.png         <-- Figure 3: Secure Throughput Analysis
