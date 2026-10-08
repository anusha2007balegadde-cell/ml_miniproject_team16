"""
PROBLEM 2 - Steady-state heat conduction (paper Section 4.2)

Poisson equation (paper Eq. 4.6 - 4.7):
    T_xx + T_yy + 1 = 0      on (-1, 1) x (-1, 1)
    T = 0                    on the boundary (soft constraint -> boundary loss)

Paper settings:
    1500 residual points, 500 boundary points
    4 hidden layers x 50 neurons, tanh, Glorot uniform init
    no output transform
    w_f = w_b = 1/2
    Adam (lr 5e-4) for 50,000 epochs, then L-BFGS until convergence
    Reference: high-accuracy numerical solution (paper: FEM / deal.II,
               here: finite differences, see src/references.py)

Run:
    python experiments/heat.py --quick
    python experiments/heat.py
    python experiments/heat.py --load
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

W_F, W_B = 0.5, 0.5


def build_model(device):
    # The paper uses NO output transform here, so the boundary condition is
    # only learned through the loss. That is why its largest errors appear
    # near the boundary. (Part 2 / Fix 1: replace None with a transform.)
    return PINN(in_dim=2, out_dim=1, hidden_width=50, n_hidden=4,
                lb=[-1.0, -1.0], ub=[1.0, 1.0], init="glorot_uniform",
                output_transform=None).to(device)


def pde_residual(model, X):
    """r = T_xx + T_yy + 1  (X must have requires_grad=True)."""
    T = model(X)
    dT = grad(T, X)
    T_x, T_y = dT[:, 0:1], dT[:, 1:2]
    T_xx = grad(T_x, X)[:, 0:1]
    T_yy = grad(T_y, X)[:, 1:2]
    return T_xx + T_yy + 1.0


def sample_boundary(rng, n):
    """n random points on the four edges of the square [-1, 1]^2."""
    side = rng.integers(0, 4, n)
    s = rng.uniform(-1, 1, n)
    x = np.where(side == 0, -1.0, np.where(side == 1, 1.0, s))
    y = np.where(side == 2, -1.0, np.where(side == 3, 1.0, s))
    return np.column_stack([x, y])


def main(argv=None):
    ap = argparse.ArgumentParser(description="PINN: 2D steady heat conduction")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lbfgs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--load", action="store_true")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args(argv)

    if not args.show:
        import matplotlib
        matplotlib.use("Agg")
    from src.plotting import plot_field_results

    adam_epochs = args.epochs or (5000 if args.quick else 50000)
    lbfgs_iters = args.lbfgs if args.lbfgs is not None else (1000 if args.quick else 10000)

    set_seed(args.seed)
    device = get_device()
    out_dir = ensure_dir(os.path.join(ROOT, "results", "heat"))
    print(f"Device: {device}")

    # ---------------- Training points ----------------
    rng = np.random.default_rng(args.seed)
    X_f = rng.uniform(-1, 1, size=(1500, 2))     # interior
    X_b = sample_boundary(rng, 500)              # boundary
    X_f_t = to_tensor(X_f, device, requires_grad=True)
    X_b_t = to_tensor(X_b, device)

    # ---------------- Reference solution + test grid ----------------
    print("Computing finite-difference reference solution ...")
    x_fd, y_fd, T_fd = heat_fd_reference(n=201)
    # use every 2nd node -> 101 x 101 = 10,201 test points (paper: 10,000)
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
        L_b = torch.mean(model(X_b_t) ** 2)      # T should be 0 on boundary
        loss = W_F * L_f + W_B * L_b
        return loss, {"L_f": L_f.item(), "L_b": L_b.item()}

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
    metrics = {
        "relative_L2_error": relative_l2(T_pred, T_ref),
        "max_abs_error": float(np.max(np.abs(T_pred - T_ref))),
        "mean_abs_error": float(np.mean(np.abs(T_pred - T_ref))),
        "mean_abs_error_on_boundary": float(np.mean(np.abs(T_pred - T_ref)[on_edge])),
        "final_train_loss": history["loss"][-1],
        "adam_epochs": adam_epochs,
        "lbfgs_max_iters": lbfgs_iters,
        "train_time_s": history["time_s"][-1],
    }
    if not args.load:
        save_json(metrics, os.path.join(out_dir, "metrics.json"))
    print("Metrics:", metrics)

    plot_field_results(Xg, Yg, T_ref, T_pred, history,
                       curves=[("L_f", "PDE train loss"),
                               ("test_pde", "PDE test loss")],
                       xlabel="x", ylabel="y", zlabel="T",
                       ref_title="Reference (finite-difference) solution",
                       path=os.path.join(out_dir, "heat_results.png"),
                       show=args.show)


if __name__ == "__main__":
    main()
