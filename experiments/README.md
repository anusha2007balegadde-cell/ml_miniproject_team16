# Physics-Informed Neural Networks for Mechanics: Reproduction and Targeted Improvements

UE24CS352A Machine Learning · Mini-Project · Problem Statement 13
PES University, Electronic City Campus

This repository has two parts:

1. **Part 1, reproduction:** the three experiments of the reference paper, *"Data Driven Solutions and Discoveries in Mechanics Using Physics Informed Neural Network"* (Zhang, Chen & Yang, Stanford CS229), re-implemented from scratch in PyTorch.
2. **Part 2, improvements:** the two improvements the paper proposes but never tests: a hard boundary condition (paper §4.2) and adaptive training-point selection (paper §5), plus a control experiment.

| # | Problem | Type | Reference solution |
|---|---------|------|--------------------|
| 1 | 1D soil consolidation (diffusion PDE) | Forward | Analytical series (paper Eq. 4.5) |
| 2 | 2D steady heat conduction (Poisson) | Forward | Our finite-difference solver |
| 3 | Damped spring-mass system | Inverse (find c, k) | Analytical (paper Eq. 4.9) |

## Results (final run, laptop CPU)

**Part 1: reproduction**

| Problem | Our result | Paper |
|---|---|---|
| Consolidation | relative L2 error 1.29 % | "perfect match" |
| Heat | relative L2 error 0.90 % (largest error on the boundary) | consistent with FEM |
| Spring (inverse) | c = 0.4115, k = 3.9904 (true 0.4, 4) | c = 0.40125, k = 3.99332 |

**Part 2: improvements**

| Experiment | Relative L2 error | Max error (t > 0) | PDE test residual |
|---|---|---|---|
| Heat, soft boundary condition (paper's method) | 0.90 % | 1.2 × 10⁻² | – |
| **Heat, hard boundary condition (Fix 1)** | **0.0040 %** | **5.5 × 10⁻⁵** | – |
| Consolidation, 500 fixed points (paper's method) | 1.29 % | 0.069 | 18.7 |
| Consolidation, +200 random points (control) | 0.84 % | 0.060 | 99.0 |
| **Consolidation, +200 RAR points (Fix 2)** | **0.79 %** | **0.032** | **0.11** |

- **Fix 1** multiplies the network output by (1 − x²)(1 − y²), so T = 0 holds exactly on the boundary. The error falls about 220 times, largely because L-BFGS converges fully once the boundary loss term is gone (4,019 iterations vs 14).
- **Fix 2** uses residual-based adaptive refinement (RAR) from DeepXDE: every 10,000 epochs it adds the 40 candidate points with the largest PDE residual. Compared with the random control, RAR halves the worst-case error and cuts the residual near t = 0 from 99 to 0.11. The average-error gain is shared with simply adding points.

## Setup

Requires Python 3.9 or newer.

```bash
git clone <repo-url>
cd pinn-mechanics
python -m venv venv
# Windows:      venv\Scripts\activate
# macOS/Linux:  source venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, if activation is blocked, first run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`.
Run every command from the repository root (the folder containing `run_all.py`).

## Running

```bash
# Show saved results instantly (no training)
python experiments/heat.py --load --show
python experiments/heat_improved.py --load --show
python experiments/consolidation_rar.py --load --show

# Reproduce everything from scratch (about 1 to 2 hours on a laptop CPU)
python run_all.py            # Part 1 (run this first)
python run_part2.py          # Part 2: Fix 1, Fix 2 and the control

# Quick check that everything runs (reduced epochs)
python run_all.py --quick
python run_part2.py --quick
```

Options available in every script: `--quick`, `--epochs N`, `--seed N`, `--load`, `--show`.
`spring_inverse.py` also accepts `--noise 0.05` (noisy observations) and `--c0`, `--k0` (initial guesses).
`consolidation_rar.py --random-add` runs the control experiment.

Running a script again overwrites its own results folder.

## Project structure

```
pinn-mechanics/
├── src/
│   ├── model.py            # PINN network: tanh MLP, Glorot init, input scaling, output transform
│   ├── trainer.py          # Adam then L-BFGS training loop with logging
│   ├── references.py       # analytical / finite-difference reference solutions
│   ├── plotting.py         # paper-style figures
│   └── utils.py            # autograd derivative helper, seeding, I/O
├── experiments/
│   ├── consolidation.py        # Part 1, problem 1
│   ├── heat.py                 # Part 1, problem 2
│   ├── spring_inverse.py       # Part 1, problem 3 (inverse)
│   ├── heat_improved.py        # Part 2, Fix 1: hard boundary condition
│   └── consolidation_rar.py    # Part 2, Fix 2: RAR (+ --random-add control)
├── results/                # one folder per run: metrics.json, history.json, model.pt, figures
├── docs/                   # slides and write-up
├── run_all.py              # runs Part 1
├── run_part2.py            # runs Part 2
└── requirements.txt
```

## Dataset

No external dataset is used, which is standard for PINNs. Training data are collocation points sampled randomly in each domain with a fixed seed. Reference solutions are the analytical solutions (consolidation, spring) and a second-order finite-difference solver (heat, 201 × 201 grid, grid-converged to about 4 × 10⁻⁶). For the inverse problem, 50 synthetic observations are generated from the exact solution with c = 0.4 and k = 4.

## Implementation notes

- Network sizes, point counts, loss weights, initialisers, learning rates and the Adam → L-BFGS schedule follow the paper.
- Inputs are scaled to [−1, 1] inside the network.
- The heat reference uses finite differences instead of the paper's deal.II finite-element solution.
- The paper's output constraints (0 ≤ p ≤ 1, T > 0) are not described in the paper and are not implemented.
- Fixed seeds give bit-identical results on the same machine; results differ slightly across machines.

## References

1. Q. Zhang, Y. Chen, Z. Yang. *Data Driven Solutions and Discoveries in Mechanics Using Physics Informed Neural Network.* Stanford CS229 report.
2. M. Raissi, P. Perdikaris, G. E. Karniadakis. *Physics-informed neural networks.* J. Comput. Phys. 378 (2019).
3. L. Lu, X. Meng, Z. Mao, G. E. Karniadakis. *DeepXDE: a deep learning library for solving differential equations.* arXiv:1907.04502 (2019). Source of RAR.
4. J. Berg, K. Nyström. *A unified deep artificial neural network approach to PDEs in complex geometries.* Neurocomputing 317 (2018). Distance-function idea behind Fix 1.

## Team

| Name | SRN |
|---|---|
| Abhay A Kora | PES2UG24AM192 |
| ______________ | ______________ |
