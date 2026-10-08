"""
PROBLEM 2 (IMPROVED) - Steady-state heat conduction with Hard Boundary Conditions

Part 2 / Fix 1:
    The paper (Section 4.2) notes that its largest heat-conduction errors occur near
    the boundary, because T = 0 on the boundary is only a soft penalty in the loss.
    It proposes multiplying the network output by a function that vanishes on the
    boundary, so the condition is satisfied exactly, but does not implement it.

    We implement that suggestion with the envelope
        T(x, y) = (1 - x^2) * (1 - y^2) * NN(x, y)
    which is exactly 0 on x = +-1 and y = +-1. The boundary loss L_b is no longer
    needed, so the loss is the PDE residual only.

    Everything else (network, 1500 interior points, Adam 50k at lr 5e-4, then
    L-BFGS, seed) is identical to Part 1, so the comparison is fair.

Run:
    python experiments/heat_improved.py --quick
    python experiments/heat_improved.py
    python experiments/heat_improved.py --load
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch

from src.model import PINN
from src.references import heat_fd_reference
from src.trainer import train
from src.utils import (ensure_dir, get_device, grad, load_json, predict,
                       relative_l2, save_json, set_seed, to_tensor)


def hard_boundary_transform(X, T_raw):
    """
    Enforces T = 0 strictly at edges x = +-1 and y = +-1.
    X: shape (N, 2) where col 0 is x and col 1 is y.
    """
    x = X[:, 0:1]
    y = X[:, 1:2]
    envelope = (1.0 - x**2) * (1.0 - y**2)
    return envelope * T_raw


def build_model(device):
    # Pass hard_boundary_transform directly to enforce hard constraints
    return PINN(in_dim=2, out_dim=1, hidden_width=50, n_hidden=4,
                lb=[-1.0, -1.0], ub=[1.0, 1.0], init="glorot_uniform",
                output_transform=hard_boundary_transform).to(device)


def pde_residual(model, X):
    """r = T_xx + T_yy + 1 (X must have requires_grad=True)."""
    T = model(X)
    dT = grad(T, X)
    T_x, T_y = dT[:, 0:1], dT[:, 1:2]
    T_xx = grad(T_x, X)[:, 0:1]
    T_yy = grad(T_y, X)[:, 1:2]
    return T_xx + T_yy + 1.0


def main(argv=None):
    ap = argparse.ArgumentParser(description="PINN: 2D steady heat conduction (Hard BC)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lbfgs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--load", action="store_true")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args(argv)

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.plotting import plot_field_results

    adam_epochs = args.epochs or (5000 if args.quick else 50000)
    lbfgs_iters = args.lbfgs if args.lbfgs is not None else (1000 if args.quick else 10000)

    set_seed(args.seed)
    device = get_device()
    
    # Save into a dedicated directory to keep Part 1 intact
    out_dir = ensure_dir(os.path.join(ROOT, "results", "heat_improved"))
    baseline_dir = os.path.join(ROOT, "results", "heat")
    print(f"Device: {device}")

    # ---------------- Training points ----------------
    rng = np.random.default_rng(args.seed)
    # With hard BCs, training concentrates entirely on interior PDE points
    X_f = rng.uniform(-1, 1, size=(1500, 2))
    X_f_t = to_tensor(X_f, device, requires_grad=True)

    # ---------------- Reference solution + test grid ----------------
    print("Computing finite-difference reference solution ...")
    x_fd, y_fd, T_fd = heat_fd_reference(n=201)
    x, y, T_ref = x_fd[::2], y_fd[::2], T_fd[::2, ::2]
    Xg, Yg = np.meshgrid(x, y)
    X_test = np.column_stack([Xg.ravel(), Yg.ravel()])
    interior = (np.abs(X_test[:, 0]) < 1) & (np.abs(X_test[:, 1]) < 1)
    X_test_int = to_tensor(X_test[interior], device, requires_grad=True)
    X_test_t = to_tensor(X_test, device)
    T_ref_t = to_tensor(T_ref.ravel()[:, None], device)

    model = build_model(device)

    def loss_fn():
        r = pde_residual(model, X_f_t)
        L_f = torch.mean(r ** 2)
        # Boundary loss is 0 by construction
        loss = L_f
        return loss, {"L_f": L_f.item(), "L_b": 0.0}

    def eval_fn():
        r = pde_residual(model, X_test_int)
        test_pde = torch.mean(r ** 2).item()
        with torch.no_grad():
            mse = torch.mean((model(X_test_t) - T_ref_t) ** 2).item()
        return {"test_pde": test_pde, "test_mse": mse}

    model_path = os.path.join(out_dir, "model.pt")
    hist_path = os.path.join(out_dir, "history.json")

    if args.load:
        model.load_state_dict(torch.load(model_path, map_location=device))
        history = load_json(hist_path)
        print(f"Loaded model from {model_path}")
    else:
        history = train(list(model.parameters()), loss_fn, adam_epochs, lr=5e-4,
                        lbfgs_iters=lbfgs_iters, log_every=500, eval_fn=eval_fn)
        torch.save(model.state_dict(), model_path)
        save_json(history, hist_path)

    T_pred = predict(model, X_test, device).reshape(Xg.shape)
    on_edge = ~interior.reshape(Xg.shape)
    abs_err = np.abs(T_pred - T_ref)

    metrics = {
        "relative_L2_error": relative_l2(T_pred, T_ref),
        "max_abs_error": float(np.max(abs_err)),
        "mean_abs_error": float(np.mean(abs_err)),
        "mean_abs_error_on_boundary": float(np.mean(abs_err[on_edge])),
        "final_train_loss": history["loss"][-1],
        "adam_epochs": adam_epochs,
        "lbfgs_max_iters": lbfgs_iters,
        "lbfgs_iters_used": history["iter"][-1] - adam_epochs,
        "train_time_s": history["time_s"][-1],
    }
    # Saved in --load mode too, so metrics.json always has the latest fields
    save_json(metrics, os.path.join(out_dir, "metrics.json"))

    print("\n--- Fix 1 Results (Hard BC) ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # Standard field plot
    plot_field_results(Xg, Yg, T_ref, T_pred, history,
                       curves=[("L_f", "PDE train loss"),
                               ("test_pde", "PDE test loss")],
                       xlabel="x", ylabel="y", zlabel="T",
                       ref_title="Reference (finite-difference) solution",
                       path=os.path.join(out_dir, "heat_results.png"),
                       show=False)

    # ---------------- Side-by-Side Comparison Plot ----------------
    baseline_model_path = os.path.join(baseline_dir, "model.pt")
    if os.path.exists(baseline_model_path):
        from experiments.heat import build_model as build_baseline_model
        base_model = build_baseline_model(device)
        base_model.load_state_dict(torch.load(baseline_model_path, map_location=device))
        T_base_pred = predict(base_model, X_test, device).reshape(Xg.shape)
        base_err = np.abs(T_base_pred - T_ref)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        vmax = max(np.max(base_err), np.max(abs_err))
        
        levels = np.linspace(0, vmax, 51)
        im0 = axes[0].contourf(Xg, Yg, base_err, levels=levels, cmap="viridis")
        axes[0].set_title("Before: soft boundary condition (paper's method)")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].contourf(Xg, Yg, abs_err, levels=levels, cmap="viridis")
        axes[1].set_title("After: hard boundary condition (same colour scale)")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        fig.colorbar(im1, ax=axes[1])

        plt.tight_layout()
        comp_path = os.path.join(out_dir, "heat_comparison_before_after.png")
        plt.savefig(comp_path, dpi=200)
        print(f"\nSaved side-by-side comparison figure: {comp_path}")

        # Second version: each panel with its own colour scale, so the
        # (much smaller) remaining error pattern of the hard-BC model is visible
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, e, title in [(axes[0], base_err, "Before: soft BC (own scale)"),
                             (axes[1], abs_err, "After: hard BC (own scale)")]:
            im = ax.contourf(Xg, Yg, e, levels=50, cmap="viridis")
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.colorbar(im, ax=ax, format="%.1e")
        plt.tight_layout()
        comp2 = os.path.join(out_dir, "heat_comparison_own_scales.png")
        plt.savefig(comp2, dpi=200)
        print(f"Saved figure: {comp2}")
    else:
        print("Part 1 model not found in results/heat/, skipping comparison figures.")

    if args.show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()