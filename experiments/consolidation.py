"""
PROBLEM 1 - One-dimensional consolidation (paper Section 4.1)

Dimensionless PDE (paper Eq. 4.4):
    p_t - p_xx = 0          0 < x < 1,  0 < t <= 0.5
    p   = 0       at x = 0  (drained top)          -> built into the output
    p_x = 0       at x = 1  (impermeable bottom)   -> boundary loss
    p   = 1       at t = 0  (initial pressure)     -> boundary loss

Paper settings:
    500 residual points, 250 points on the x-boundary, 125 on the t-boundary
    5 hidden layers x 50 neurons, tanh, Glorot normal init
    output = x * network(x, t)    (satisfies p(0, t) = 0 automatically)
    w_f = w_b = 1/4
    Adam 50,000 epochs (lr 1e-3), then L-BFGS until convergence

Run:
    python experiments/consolidation.py --quick     (fast test, a few minutes)
    python experiments/consolidation.py             (full paper settings)
    python experiments/consolidation.py --load      (re-plot a saved model)
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch

from src.model import PINN
from src.references import consolidation_exact
from src.trainer import train
from src.utils import (ensure_dir, get_device, grad, load_json, predict,
                       relative_l2, save_json, set_seed, to_tensor)

T_MAX = 0.5          # dimensionless final time (paper: t_max = 0.5)
W_F, W_B = 0.25, 0.25


def build_model(device):
    # Output transform from the paper: multiply by x so p(0, t) = 0 exactly
    def times_x(X, out):
        return X[:, 0:1] * out

    return PINN(in_dim=2, out_dim=1, hidden_width=50, n_hidden=5,
                lb=[0.0, 0.0], ub=[1.0, T_MAX], init="glorot_normal",
                output_transform=times_x).to(device)


def pde_residual(model, X):
    """r = p_t - p_xx  (X must have requires_grad=True)."""
    p = model(X)
    dp = grad(p, X)                 # [p_x, p_t]
    p_x, p_t = dp[:, 0:1], dp[:, 1:2]
    p_xx = grad(p_x, X)[:, 0:1]
    return p_t - p_xx


def main(argv=None):
    ap = argparse.ArgumentParser(description="PINN: 1D consolidation")
    ap.add_argument("--quick", action="store_true", help="short run for testing")
    ap.add_argument("--epochs", type=int, default=None, help="Adam epochs")
    ap.add_argument("--lbfgs", type=int, default=None, help="max L-BFGS iterations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--load", action="store_true", help="load saved model, skip training")
    ap.add_argument("--show", action="store_true", help="open the figure window")
    args = ap.parse_args(argv)

    if not args.show:
        import matplotlib
        matplotlib.use("Agg")
    from src.plotting import plot_field_results

    adam_epochs = args.epochs or (5000 if args.quick else 50000)
    lbfgs_iters = args.lbfgs if args.lbfgs is not None else (1000 if args.quick else 10000)

    set_seed(args.seed)
    device = get_device()
    out_dir = ensure_dir(os.path.join(ROOT, "results", "consolidation"))
    print(f"Device: {device}")

    # ---------------- Training points (our "dataset") ----------------
    rng = np.random.default_rng(args.seed)
    n_f, n_bx, n_bt = 500, 250, 125
    X_f = np.column_stack([rng.uniform(0, 1, n_f), rng.uniform(0, T_MAX, n_f)])
    X_bx = np.column_stack([np.ones(n_bx), rng.uniform(0, T_MAX, n_bx)])   # x = 1
    X_bt = np.column_stack([rng.uniform(0, 1, n_bt), np.zeros(n_bt)])      # t = 0

    X_f_t = to_tensor(X_f, device, requires_grad=True)
    X_bx_t = to_tensor(X_bx, device, requires_grad=True)
    X_bt_t = to_tensor(X_bt, device)

    # ---------------- Test set (as in the paper) ----------------
    # Paper: "The test set will have 10011 points". We draw them randomly from
    # the domain (0,1) x (0, t_max), like the training points. A separate random
    # generator is used so the TRAINING points are exactly the same as before.
    rng_test = np.random.default_rng(args.seed + 1000)
    n_test = 10011
    X_test = np.column_stack([rng_test.uniform(0, 1, n_test),
                              rng_test.uniform(0, T_MAX, n_test)])
    p_test_exact = consolidation_exact(X_test[:, 0], X_test[:, 1])[:, None]
    X_test_g = to_tensor(X_test, device, requires_grad=True)   # for PDE test loss
    X_test_t = to_tensor(X_test, device)                       # for prediction
    p_test_exact_t = to_tensor(p_test_exact, device)

    # ---------------- Plotting grid ----------------
    # Regular grid, only used to draw the figures.
    x = np.linspace(0, 1, 101)
    t = np.linspace(0, T_MAX, 101)
    Xg, Tg = np.meshgrid(x, t)                       # shape (n_t, n_x)
    X_grid = np.column_stack([Xg.ravel(), Tg.ravel()])
    p_grid_exact = consolidation_exact(Xg, Tg)

    model = build_model(device)

    # ---------------- Physics-informed loss (paper Eq. 3.3) ----------------
    def loss_fn():
        r = pde_residual(model, X_f_t)
        L_f = torch.mean(r ** 2)                       # PDE residual loss

        p_bx = model(X_bx_t)
        p_x_bx = grad(p_bx, X_bx_t)[:, 0:1]            # p_x at x = 1 -> 0
        p_bt = model(X_bt_t)                           # p at t = 0   -> 1
        b_res = torch.cat([p_x_bx, p_bt - 1.0])
        L_b = torch.mean(b_res ** 2)                   # boundary + initial loss

        loss = W_F * L_f + W_B * L_b
        return loss, {"L_f": L_f.item(), "L_b": L_b.item()}

    def eval_fn():
        r = pde_residual(model, X_test_g)
        test_pde = torch.mean(r ** 2).item()
        with torch.no_grad():
            mse = torch.mean((model(X_test_t) - p_test_exact_t) ** 2).item()
        return {"test_pde": test_pde, "test_mse": mse}

    model_path = os.path.join(out_dir, "model.pt")
    hist_path = os.path.join(out_dir, "history.json")

    if args.load:
        model.load_state_dict(torch.load(model_path, map_location=device))
        history = load_json(hist_path)
        print(f"Loaded model from {model_path}")
    else:
        history = train(list(model.parameters()), loss_fn, adam_epochs, lr=1e-3,
                        lbfgs_iters=lbfgs_iters, log_every=500, eval_fn=eval_fn)
        torch.save(model.state_dict(), model_path)
        save_json(history, hist_path)

    # ---------------- Results ----------------
    # 1) Accuracy on the random test set (what the paper reports)
    p_test_pred = predict(model, X_test, device)
    test_err = np.abs(p_test_pred - p_test_exact)

    # 2) Accuracy on the plotting grid. At t = 0 the exact solution jumps from
    #    0 (at x = 0) to 1 (for x > 0). No smooth network can represent a jump,
    #    so we report grid errors both with and without the t = 0 row.
    p_grid_pred = predict(model, X_grid, device).reshape(Xg.shape)
    grid_err = np.abs(p_grid_pred - p_grid_exact)
    after_t0 = Tg > 0

    metrics = {
        "test_set": f"{n_test} random points in (0,1) x (0,{T_MAX})",
        "relative_L2_error": relative_l2(p_test_pred, p_test_exact),
        "max_abs_error": float(test_err.max()),
        "mean_abs_error": float(test_err.mean()),
        "grid_relative_L2_error_t_gt_0": relative_l2(p_grid_pred[after_t0],
                                                     p_grid_exact[after_t0]),
        "grid_max_abs_error_t_gt_0": float(grid_err[after_t0].max()),
        "grid_max_abs_error_incl_t0_jump": float(grid_err.max()),
        "final_train_loss": history["loss"][-1],
        "final_test_pde_loss": history["test_pde"][-1],
        "adam_epochs": adam_epochs,
        "lbfgs_max_iters": lbfgs_iters,
        "train_time_s": history["time_s"][-1],
    }
    if not args.load:
        save_json(metrics, os.path.join(out_dir, "metrics.json"))
    print("Metrics:", metrics)

    plot_field_results(Xg, Tg, p_grid_exact, p_grid_pred, history,
                       curves=[("L_f", "PDE train loss"),
                               ("test_pde", "PDE test loss"),
                               ("test_mse", "Mean square error vs exact")],
                       xlabel="x", ylabel="t", zlabel="p",
                       ref_title="Reference (analytical) solution",
                       path=os.path.join(out_dir, "consolidation_results.png"),
                       show=args.show,
                       err_mask=~after_t0,
                       err_title="Absolute error (t > 0; t = 0 jump excluded)")


if __name__ == "__main__":
    main()
