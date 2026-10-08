"""
PROBLEM 1 (IMPROVED) - 1D Consolidation with Residual-based Adaptive Refinement (RAR)

Part 2 / Fix 2:
    In our Part 1 reproduction, the 500 fixed random collocation points were too
    sparse in the thin layer near t = 0, where the solution changes very sharply.
    There the PDE test loss grew to ~18.7, even though the overall error was small.

    The paper's conclusion suggests a "training points selection strategy similar to
    the adaptive re-meshing". We implement Residual-based Adaptive Refinement (RAR),
    the method of Lu et al., "DeepXDE: A deep learning library for solving
    differential equations" (reference [24] of the assigned paper):

        repeat 5 times:
            train with Adam for 10,000 epochs
            sample 5,000 candidate points, compute their PDE residual
            add the 40 candidates with the LARGEST residual to the training set
        finish with L-BFGS

    Same total budget as Part 1 (50,000 Adam epochs + L-BFGS), 500 + 200 = 700 points.

Control experiment (--random-add):
    Identical schedule and identical number of added points, but the 40 new points
    per cycle are chosen at RANDOM instead of by largest residual. This separates
    "smarter placement" from "simply more points".

Run:
    python experiments/consolidation_rar.py --quick            (test)
    python experiments/consolidation_rar.py                    (RAR, full)
    python experiments/consolidation_rar.py --random-add       (control, full)
    python experiments/consolidation_rar.py --load             (re-plot saved RAR model)
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

T_MAX = 0.5
W_F, W_B = 0.25, 0.25


def build_model(device):
    def times_x(X, out):
        return X[:, 0:1] * out

    return PINN(in_dim=2, out_dim=1, hidden_width=50, n_hidden=5,
                lb=[0.0, 0.0], ub=[1.0, T_MAX], init="glorot_normal",
                output_transform=times_x).to(device)


def pde_residual(model, X):
    """r = p_t - p_xx (X must have requires_grad=True)."""
    p = model(X)
    dp = grad(p, X)
    p_x, p_t = dp[:, 0:1], dp[:, 1:2]
    p_xx = grad(p_x, X)[:, 0:1]
    return p_t - p_xx


def merge_histories(history_list):
    """
    Each call to train() restarts its iteration counter and timer at 0.
    Shift every segment so iterations and time run continuously.
    """
    merged = {k: [] for k in history_list[0].keys()}
    it_offset, time_offset = 0, 0.0
    for h in history_list:
        for k in merged:
            if k == "iter":
                merged[k].extend([i + it_offset for i in h[k]])
            elif k == "time_s":
                merged[k].extend([s + time_offset for s in h[k]])
            else:
                merged[k].extend(h[k])
        it_offset = merged["iter"][-1]
        time_offset = merged["time_s"][-1]
    return merged


def main(argv=None):
    ap = argparse.ArgumentParser(description="PINN: 1D consolidation with RAR")
    ap.add_argument("--quick", action="store_true", help="short run for testing")
    ap.add_argument("--epochs", type=int, default=None, help="total Adam epochs")
    ap.add_argument("--lbfgs", type=int, default=None, help="max L-BFGS iterations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--random-add", action="store_true",
                    help="control: add random points instead of largest-residual points")
    ap.add_argument("--load", action="store_true", help="load saved model")
    ap.add_argument("--show", action="store_true", help="open the figure windows")
    args = ap.parse_args(argv)

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.plotting import plot_field_results

    adam_epochs = args.epochs or (5000 if args.quick else 50000)
    lbfgs_iters = args.lbfgs if args.lbfgs is not None else (1000 if args.quick else 10000)

    rar_cycles = 5
    epochs_per_cycle = adam_epochs // rar_cycles
    num_candidates = 5000
    points_per_cycle = 40   # 40 x 5 = 200 added points in total

    mode = "random_add" if args.random_add else "rar"
    mode_label = "Random-add control" if args.random_add else "RAR"

    set_seed(args.seed)
    device = get_device()
    folder = "consolidation_rar_random_control" if args.random_add else "consolidation_rar"
    out_dir = ensure_dir(os.path.join(ROOT, "results", folder))
    baseline_dir = os.path.join(ROOT, "results", "consolidation")
    print(f"Device: {device} | Mode: {mode_label}")

    # ---------------- Initial training points (identical to Part 1) ----------------
    rng = np.random.default_rng(args.seed)
    n_f, n_bx, n_bt = 500, 250, 125
    X_f = np.column_stack([rng.uniform(0, 1, n_f), rng.uniform(0, T_MAX, n_f)])
    X_bx = np.column_stack([np.ones(n_bx), rng.uniform(0, T_MAX, n_bx)])   # x = 1
    X_bt = np.column_stack([rng.uniform(0, 1, n_bt), np.zeros(n_bt)])      # t = 0
    X_f_initial = X_f.copy()

    X_f_t = to_tensor(X_f, device, requires_grad=True)
    X_bx_t = to_tensor(X_bx, device, requires_grad=True)
    X_bt_t = to_tensor(X_bt, device)

    # ---------------- Test set (identical to Part 1) ----------------
    rng_test = np.random.default_rng(args.seed + 1000)
    n_test = 10011
    X_test = np.column_stack([rng_test.uniform(0, 1, n_test),
                              rng_test.uniform(0, T_MAX, n_test)])
    p_test_exact = consolidation_exact(X_test[:, 0], X_test[:, 1])[:, None]
    X_test_g = to_tensor(X_test, device, requires_grad=True)
    X_test_t = to_tensor(X_test, device)
    p_test_exact_t = to_tensor(p_test_exact, device)

    # ---------------- Plotting grid ----------------
    x = np.linspace(0, 1, 101)
    t = np.linspace(0, T_MAX, 101)
    Xg, Tg = np.meshgrid(x, t)
    X_grid = np.column_stack([Xg.ravel(), Tg.ravel()])
    p_grid_exact = consolidation_exact(Xg, Tg)
    after_t0 = Tg > 0

    model = build_model(device)

    # loss_fn reads X_f_t when it is called, so it automatically uses the
    # enlarged point set after each refinement step.
    def loss_fn():
        r = pde_residual(model, X_f_t)
        L_f = torch.mean(r ** 2)

        p_bx = model(X_bx_t)
        p_x_bx = grad(p_bx, X_bx_t)[:, 0:1]
        p_bt = model(X_bt_t)
        b_res = torch.cat([p_x_bx, p_bt - 1.0])
        L_b = torch.mean(b_res ** 2)

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
    pts_file = os.path.join(out_dir, "collocation_points.npz")

    if args.load:
        model.load_state_dict(torch.load(model_path, map_location=device))
        history = load_json(hist_path)
        saved = np.load(pts_file)
        X_f_initial, X_f = saved["initial"], saved["final"]
        lbfgs_used = None
        print(f"Loaded model from {model_path}")
    else:
        history_list = []
        print(f"\n--- {mode_label}: {rar_cycles} cycles x {epochs_per_cycle} Adam epochs ---")

        for cycle in range(rar_cycles):
            print(f"\n>>> Cycle {cycle + 1}/{rar_cycles} | collocation points: {X_f.shape[0]}")
            h = train(list(model.parameters()), loss_fn, epochs_per_cycle, lr=1e-3,
                      lbfgs_iters=0, log_every=500, eval_fn=eval_fn)
            history_list.append(h)

            # Candidate pool, sampled uniformly over the domain
            cand = np.column_stack([rng.uniform(0, 1, num_candidates),
                                    rng.uniform(0, T_MAX, num_candidates)])
            cand_t = to_tensor(cand, device, requires_grad=True)
            sq_res = (pde_residual(model, cand_t) ** 2).detach().cpu().numpy().ravel()

            if args.random_add:
                chosen = np.arange(points_per_cycle)          # candidates are already random
            else:
                chosen = np.argsort(sq_res)[-points_per_cycle:]   # largest residuals

            X_f = np.vstack([X_f, cand[chosen]])
            X_f_t = to_tensor(X_f, device, requires_grad=True)
            print(f"Added {points_per_cycle} points ({mode_label}). "
                  f"Max squared residual among candidates: {sq_res.max():.4e}")

        print(f"\n>>> Final L-BFGS refinement (up to {lbfgs_iters} iterations) ...")
        h_lbfgs = train(list(model.parameters()), loss_fn, 0, lr=1e-3,
                        lbfgs_iters=lbfgs_iters, log_every=500, eval_fn=eval_fn)
        lbfgs_used = h_lbfgs["iter"][-1]
        history_list.append(h_lbfgs)

        history = merge_histories(history_list)
        torch.save(model.state_dict(), model_path)
        save_json(history, hist_path)
        np.savez(pts_file, initial=X_f_initial, final=X_f)

    # ---------------- Metrics (same definitions as Part 1) ----------------
    p_test_pred = predict(model, X_test, device)
    test_err = np.abs(p_test_pred - p_test_exact)
    p_grid_pred = predict(model, X_grid, device).reshape(Xg.shape)
    grid_err = np.abs(p_grid_pred - p_grid_exact)

    metrics = {
        "mode": mode,
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
        "initial_collocation_points": int(X_f_initial.shape[0]),
        "total_collocation_points": int(X_f.shape[0]),
        "refinement_cycles": rar_cycles,
        "points_added_per_cycle": points_per_cycle,
        "adam_epochs": epochs_per_cycle * rar_cycles,
        "lbfgs_max_iters": lbfgs_iters,
        "lbfgs_iters_used": lbfgs_used,
        "train_time_s": history["time_s"][-1],
    }
    if not args.load:
        save_json(metrics, os.path.join(out_dir, "metrics.json"))

    print(f"\n--- Fix 2 results ({mode_label}) ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # ---------------- Figure 1: standard results figure ----------------
    plot_field_results(Xg, Tg, p_grid_exact, p_grid_pred, history,
                       curves=[("L_f", "PDE train loss"),
                               ("test_pde", "PDE test loss"),
                               ("test_mse", "Mean square error vs exact")],
                       xlabel="x", ylabel="t", zlabel="p",
                       ref_title="Reference (analytical) solution",
                       path=os.path.join(out_dir, "consolidation_results.png"),
                       show=False,
                       err_mask=~after_t0,
                       err_title="Absolute error (t > 0; t = 0 jump excluded)")

    # ---------------- Figure 2: collocation points ----------------
    added = X_f[X_f_initial.shape[0]:]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(X_f_initial[:, 0], X_f_initial[:, 1], s=10, c="royalblue", alpha=0.7)
    axes[0].set_title(f"Part 1 baseline: {X_f_initial.shape[0]} random points")
    axes[1].scatter(X_f_initial[:, 0], X_f_initial[:, 1], s=10, c="royalblue",
                    alpha=0.35, label=f"original ({X_f_initial.shape[0]})")
    axes[1].scatter(added[:, 0], added[:, 1], s=16, c="crimson",
                    label=f"added by {mode_label} ({added.shape[0]})")
    if args.random_add:
        axes[1].set_title("Control: points added at random")
    else:
        axes[1].set_title("RAR: added points cluster near t = 0")
    axes[1].legend(loc="upper right")
    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, T_MAX)
    fig.tight_layout()
    pts_path = os.path.join(out_dir, "consolidation_points_distribution.png")
    fig.savefig(pts_path, dpi=200)
    print(f"Saved figure: {pts_path}")

    # ---------------- Figure 3: before / after error, same colour scale ----------------
    baseline_model_path = os.path.join(baseline_dir, "model.pt")
    if os.path.exists(baseline_model_path):
        base_model = build_model(device)
        base_model.load_state_dict(torch.load(baseline_model_path, map_location=device))
        base_err = np.abs(predict(base_model, X_grid, device).reshape(Xg.shape) - p_grid_exact)
        errs = [np.ma.masked_where(~after_t0, base_err),
                np.ma.masked_where(~after_t0, grid_err)]
        vmax = max(e.max() for e in errs)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        titles = ["Before: 500 fixed points (paper's method)",
                  f"After: {mode_label} ({X_f.shape[0]} points)"]
        for ax, e, title in zip(axes, errs, titles):
            pc = ax.pcolormesh(Xg, Tg, e, cmap="jet", vmin=0, vmax=vmax, shading="auto")
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("t")
            fig.colorbar(pc, ax=ax, label="|error| (t > 0)")
        fig.tight_layout()
        comp_path = os.path.join(out_dir, "consolidation_comparison_before_after.png")
        fig.savefig(comp_path, dpi=200)
        print(f"Saved figure: {comp_path}")
    else:
        print("Part 1 model not found in results/consolidation/, skipping comparison figure.")

    if args.show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
