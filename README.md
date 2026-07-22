# HELP-RL

**Real-Time Policy Optimization with a New Hierarchical Decoder for Heterogeneous and Stochastic Disaster Response Logistics**

[![Paper](https://img.shields.io/badge/Paper-DOI%2010.1016%2Fj.cor.2026.107604-orange)](https://doi.org/10.1016/j.cor.2026.107604)
[![Journal](https://img.shields.io/badge/Journal-Computers%20%26%20Operations%20Research%2C%20Vol.%20195-blue)](https://www.sciencedirect.com/journal/computers-and-operations-research/vol/195/suppl/C)
[![License](https://img.shields.io/badge/License-CC0--1.0-lightgrey)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9-blue)](Dockerfile)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2011.8-red)](Dockerfile)

This repository contains the official code accompanying the paper **"HELP-RL: Real-Time Policy Optimization with New Hierarchical Decoder for Heterogeneous and Stochastic Disaster Response Logistics,"** accepted for publication in *Computers & Operations Research*.

**Authors:** Hadi Aghazadeh, Xin Wang

📄 **Paper:** https://doi.org/10.1016/j.cor.2026.107604 ([Volume 195](https://www.sciencedirect.com/journal/computers-and-operations-research/vol/195/suppl/C), November 2026, Article 107604)

---

## Table of Contents

- [Overview](#overview)
- [Abstract](#abstract)
- [Key Contributions](#key-contributions)
- [Method](#method)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Requirements](#requirements)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Overview

**HELP-RL** (Humanitarian Emergency Logistics Planning with Reinforcement Learning) is a reinforcement-learning framework for **real-time disaster response logistics**. Unlike most prior work, which focuses on pre-disaster planning, HELP-RL targets the dynamic, evolving decision-making required *during* an emergency, where fleets are heterogeneous, demand types are diverse, and travel times are uncertain.

The framework learns a dispatch policy with an attention-based encoder and a novel **hierarchical decoder** built on **dilated causal convolutions (DCC)**, trained end-to-end with the REINFORCE policy-gradient algorithm and a learned critic baseline.

## Abstract

Disaster response logistics plays a vital role in reducing existential risk in relief operations by ensuring that essential resources reach affected communities promptly during emergencies. Although its significance is widely recognized, existing studies tend to focus primarily on pre-disaster planning, leaving notable gaps in addressing the complex, dynamic, and unpredictable challenges of real-time disaster response.

This paper introduces a new framework called **HELP-RL** (Humanitarian Emergency Logistics Planning with Reinforcement Learning) to address these challenges, grouped into three main categories: **heterogeneous fleet types**, **diverse demand types**, and **stochastic travel times**. Together, these reflect the real-world complexities of disaster response logistics, all of which require real-time management.

The HELP-RL framework incorporates a unique hierarchical decoder to manage the intricate action space, paired with an efficiently designed attention-based policy optimizer using the REINFORCE algorithm. This fully integrated model is built to adapt dynamically to changing disaster conditions, providing reliable and real-time solutions. To assess its performance, the framework is tested using both synthetic and real-world data from wildfire response logistics, drawing on actual data to replicate realistic scenarios. The findings show that HELP-RL produces adaptable and optimal policies that effectively handle rapid changes, surpassing other reinforcement learning approaches in terms of both efficiency and practical applicability.

## Key Contributions

- **Hierarchical action decoding** — a two-tier decoder that first decides whether to dispatch from the main hub, and then, for help hubs, jointly resolves *which hub*, *which vehicle type*, and *which demand type* to serve — collapsing an otherwise combinatorial action space into a tractable, differentiable policy.
- **Dilated causal convolution (DCC) stack** — replaces recurrent/attention-only decoding over help hubs with a stack of gated dilated convolutions (WaveNet-style), giving a wide receptive field over hubs with residual and skip connections at low computational cost.
- **Explicit stochastic travel-time modeling** — travel time estimates and their uncertainty are computed from node coordinates and injected directly into the decoder's state, so the policy reasons about *risk*, not just distance.
- **Heterogeneous fleets and multi-type demand** — the model natively supports multiple vehicle types with different capacities/costs and multiple demand categories with different importance weights, matching real disaster-relief operations.
- **Actor–critic REINFORCE training** — a shared encoder feeds both a policy head (hierarchical decoder) and a critic head, using the critic's value estimate as a variance-reducing baseline for the policy gradient.
- **Validated on real wildfire-response data** — in addition to synthetic benchmarks, the framework is evaluated against real-world wildfire disaster logistics data.

## Method

At a high level, the model (`HierarchicalReinforce` in [`DCConv_reinforce.py`](DCConv_reinforce.py)) is composed of four parts:

1. **Encoder (`SimpleEncoder`)** — a shared feed-forward network that embeds each node (the main hub and every help hub) from its raw features (coordinates + per-node demand vector) into a hidden representation.
2. **Global self-attention** — a multi-head attention layer mixes information across all nodes so each embedding is aware of the full problem instance, not just its local features.
3. **Hierarchical decoder (`DilatedCausalConvolutionHierarchicalDecoder`)**:
   - Fuses the current dynamic state (current location + remaining vehicle capacities) and travel-time/uncertainty features into every node embedding.
   - Scores the **main hub** directly with a pointer head.
   - Scores **help hubs** through a stack of `DilatedCausalConvolutionBlock`s (gated tanh/sigmoid activations, residual + skip connections, exponentially increasing dilation), producing a hub-selection distribution and, per hub, a vehicle-type × demand-type distribution.
   - Combines these into one flat action distribution: `[main_hub] + [help_hub × vehicle_type × demand_type]`.
4. **Critic (`CriticNetwork`)** — estimates the state value from pooled node embeddings and the dynamic state, used as the REINFORCE baseline.

Training (see [`train_DCConv_reinforce.py`](train_DCConv_reinforce.py)) rolls out full episodes in a batched environment, accumulates log-probabilities and critic values at every step, then optimizes a combined **policy-gradient + critic (MSE) loss** with gradient clipping.

## Repository Structure

```
.
├── DCConv_reinforce.py        # Model definition: encoder, attention, hierarchical DCC decoder, critic
├── train_DCConv_reinforce.py  # Training loop (REINFORCE + critic baseline)
├── config_help.yaml           # All environment, model, and training hyperparameters
├── Dockerfile                 # CUDA 11.8 + Python 3.9 environment for training
├── LICENSE                    # CC0 1.0 Universal
└── README.md
```

> **Note:** `train_DCConv_reinforce.py` imports the simulation environment as `from HELPRL_Env import HELP_RL_ENV`. This environment module defines the disaster-logistics simulator (hub generation, demand sampling, stochastic travel times, and reward computation) referenced in the paper. If it is not yet present in this repository, add your `HELPRL_Env.py` implementation (or the release matching the paper) alongside the files above before running training.

## Installation

### Option A — Docker (recommended, matches the paper's environment)

```bash
git clone https://github.com/hadiagha/helprl-paper.git
cd helprl-paper
docker build -t help-rl .
docker run --gpus all -it -v $(pwd):/workspace -w /workspace help-rl bash
```

The provided [`Dockerfile`](Dockerfile) builds on `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04`, installs Python 3.9, and installs PyTorch (CUDA 11.8 build) along with `matplotlib`, `numpy`, `pytz`, and `PyYAML`.

### Option B — Local environment

```bash
git clone https://github.com/hadiagha/helprl-paper.git
cd helprl-paper
python3.9 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install matplotlib numpy pytz PyYAML
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

CPU-only machines can omit the CUDA index URL and install the standard `torch` build instead; the code automatically falls back to CPU (`torch.device('cuda' if torch.cuda.is_available() else 'cpu')`).

## Configuration

All problem, model, and training settings live in [`config_help.yaml`](config_help.yaml) under `hierarchical_reinforce_training`:

| Parameter | Description |
|---|---|
| `num_help_hubs` | Number of help hubs (delivery/demand nodes) in the instance |
| `num_vehicle_types` | Number of heterogeneous vehicle types in the fleet |
| `num_demand_types` | Number of distinct demand/resource categories |
| `cost_per_time` | Operating cost per unit time, per vehicle type |
| `full_capacities` | `(num_vehicle_types × num_demand_types)` matrix of vehicle capacity per demand type |
| `travel_time_mean`, `travel_time_std` | Parameters of the stochastic travel-time model |
| `demand_scaler` | Scaling factor applied to sampled demand magnitudes |
| `demand_importance` | Relative priority/weight of each demand type |
| `demand_cost` | `(num_vehicle_types × num_demand_types)` matrix of delivery cost per unit demand |
| `batch_size` | Number of parallel problem instances per training step |
| `hidden_dim` | Hidden dimension used throughout the encoder/decoder/critic |
| `encoder_layers` | Number of layers in the feed-forward node encoder |
| `n_heads` | Number of attention heads in the global self-attention layer |
| `num_DCC_layers` | Number of dilated causal convolution blocks in the decoder |
| `lr` | Adam learning rate |
| `num_epochs` | Number of training epochs (episodes) |
| `max_grad_norm` | Gradient clipping norm |
| `critic_loss_weight` | Weight of the critic MSE loss relative to the policy loss |
| `print_interval` | Logging frequency, in epochs |

Edit this file to reproduce different problem sizes or ablations reported in the paper.

## Usage

Train a model with the default configuration:

```bash
python train_DCConv_reinforce.py
```

This will:
1. Load `config_help.yaml`.
2. Instantiate the `HELP_RL_ENV` simulation environment and the `HierarchicalReinforce` model.
3. Run the REINFORCE training loop for `num_epochs`, logging loss and average reward to both the console and a timestamped log file under `help_logs/hierarchical_reinforce_DCConv/`.
4. Save the final model weights to `hierarchical_reinforce_DCConv.pth`.

To use a different configuration file, edit the `load_config()` call at the top of `train_DCConv_reinforce.py` (or point it to your own YAML path).

## Requirements

- Python 3.9
- PyTorch (CUDA 11.8 build used in the Dockerfile; CPU or other CUDA versions also work)
- numpy
- PyYAML
- matplotlib
- pytz

## Citation

If you use this code or build on HELP-RL in your research, please cite the paper:

```bibtex
@article{aghazadeh_help_rl,
  title   = {HELP-RL: Real-Time Policy Optimization with New Hierarchical Decoder
             for Heterogeneous and Stochastic Disaster Response Logistics},
  author  = {Aghazadeh, Hadi and Wang, Xin},
  journal = {Computers \& Operations Research},
  volume  = {195},
  pages   = {107604},
  year    = {2026},
  month   = {November},
  doi     = {10.1016/j.cor.2026.107604},
  url     = {https://doi.org/10.1016/j.cor.2026.107604},
  publisher = {Elsevier}
}
```

## License

This project is released under the **CC0 1.0 Universal** license — see [LICENSE](LICENSE) for details.

## Contact

For questions about the paper or code, please open an issue in this repository or reach out to the authors, **Hadi Aghazadeh** and **Xin Wang**.
