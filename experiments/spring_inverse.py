"""
PROBLEM 3 - Inverse problem: damped spring-mass system (paper Section 4.3)

ODE (paper Eq. 4.8), with m = 1:
    u'' + c u' + k u = 0,    u(0) = 1,  u'(0) = 1

The damping c and stiffness k are UNKNOWN. The PINN learns them, together
with u(t), from 50 observations generated with the true values c = 0.4, k = 4.

Paper settings:
    50 equispaced observation points (data loss)
    100 random residual points (ODE loss)
    initial-condition loss at t = 0
    10,000 test points
    3 hidden layers x 32 neurons, tanh, Glorot uniform init
    w_f = w_b = w_d = 1/4
    Adam only, 100,000 epochs, lr 5e-4
    Paper result: c = 0.40125, k = 3.99332

Run:
    python experiments/spring_inverse.py --quick
    python experiments/spring_inverse.py
    python experiments/spring_inverse.py --noise 0.05    (optional: 5% noisy data)
    python experiments/spring_inverse.py --load
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn as nn

from src.model import PINN
from src.references import spring_exact
from src.trainer import train
from src.utils import (ensure_dir, get_device, grad, load_json, predict,
                       relative_l2, save_json, set_seed, to_tensor)

T_END = 15.0
M, U0, V0 = 1.0, 1.0, 1.0
C_TRUE, K_TRUE = 0.4, 4.0
W_F = W_B = W_D = 0.25


class InversePINN(nn.Module):
    """The network u(t) plus the two unknown physical parameters c and k."""

    def __init__(self, c_init=1.0, k_init=1.0):
        super().__init__()
        self.net = PINN(in_dim=1, out_dim=1, hidden_width=32, n_hidden=3,
                        lb=[0.0], ub=[T_END], init="glorot_uniform")
        # nn.Parameter -> the optimizer updates c and k like any weight
        self.c = nn.Parameter(torch.tensor(float(c_init)))
        self.k = nn.Parameter(torch.tensor(float(k_init)))

    def forward(self, t):
        return self.net(t)


def ode_residual(model, t):
    """r = m u'' + c u' + k u  (t must have requires_grad=True)."""
    u = model(t)
    u_t = grad(u, t)
    u_tt = grad(u_t, t)
    return M * u_tt + model.c * u_t + model.k * u


def main(argv=None):
    ap = argparse.ArgumentParser(description="PINN: inverse spring-mass problem")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.0,
                    help="relative noise level added to the observations")
    ap.add_argument("--c0", type=float, default=1.0, help="initial guess for c")
    ap.add_argument("--k0", type=float, default=1.0, help="initial guess for k")
    ap.add_argument("--load", action="store_true")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args(argv)

    if not args.show:
        import matplotlib
        matplotlib.use("Agg")
    from src.plotting import plot_spring_results

    adam_epochs = args.epochs or (20000 if args.quick else 100000)

    set_seed(args.seed)
    device = get_device()
    tag = "spring_inverse" if args.noise == 0 else f"spring_inverse_noise{args.noise}"
    out_dir = ensure_dir(os.path.join(ROOT, "results", tag))
    print(f"Device: {device}")

    # ---------------- Data: synthetic observations ----------------
    rng = np.random.default_rng(args.seed)
    t_obs = np.linspace(0, T_END, 50)[:, None]
    u_obs = spring_exact(t_obs, M, C_TRUE, K_TRUE, U0, V0)
    if args.noise > 0:
        u_obs = u_obs + args.noise * np.std(u_obs) * rng.standard_normal(u_obs.shape)

    t_f = rng.uniform(0, T_END, size=(100, 1))     # residual points
    t_f_t = to_tensor(t_f, device, requires_grad=True)
    t_0_t = to_tensor([[0.0]], device, requires_grad=True)   # initial condition
    t_obs_t = to_tensor(t_obs, device)
    u_obs_t = to_tensor(u_obs, device)

    t_test = np.linspace(0, T_END, 10000)[:, None]
    u_true = spring_exact(t_test, M, C_TRUE, K_TRUE, U0, V0)
    t_test_t = to_tensor(t_test, device, requires_grad=True)

    model = InversePINN(args.c0, args.k0).to(device)

    def loss_fn():
        r = ode_residual(model, t_f_t)
        L_f = torch.mean(r ** 2)                       # physics (ODE) loss

        u0 = model(t_0_t)
        v0 = grad(u0, t_0_t)
        L_b = torch.mean((u0 - U0) ** 2 + (v0 - V0) ** 2)   # initial conditions

        L_d = torch.mean((model(t_obs_t) - u_obs_t) ** 2)    # data loss

        loss = W_F * L_f + W_B * L_b + W_D * L_d
        return loss, {"L_f": L_f.item(), "L_b": L_b.item(), "L_d": L_d.item(),
                      "c": model.c.item(), "k": model.k.item()}

    def eval_fn():
        r = ode_residual(model, t_test_t)
        return {"test_pde": torch.mean(r ** 2).item()}

    model_path = os.path.join(out_dir, "model.pt")
    hist_path = os.path.join(out_dir, "history.json")

    if args.load:
        model.load_state_dict(torch.load(model_path, map_location=device))
        history = load_json(hist_path)
        print(f"Loaded model from {model_path}")
    else:
        history = train(list(model.parameters()), loss_fn, adam_epochs, lr=5e-4,
                        lbfgs_iters=0, log_every=200, eval_fn=eval_fn)
        torch.save(model.state_dict(), model_path)
        save_json(history, hist_path)

    u_pred = predict(model, t_test, device)
    c_est, k_est = model.c.item(), model.k.item()
    metrics = {
        "c_identified": c_est,
        "k_identified": k_est,
        "c_true": C_TRUE,
        "k_true": K_TRUE,
        "c_error_percent": abs(c_est - C_TRUE) / C_TRUE * 100,
        "k_error_percent": abs(k_est - K_TRUE) / K_TRUE * 100,
        "relative_L2_error_u": relative_l2(u_pred, u_true),
        "noise_level": args.noise,
        "adam_epochs": adam_epochs,
        "train_time_s": history["time_s"][-1],
    }
    if not args.load:
        save_json(metrics, os.path.join(out_dir, "metrics.json"))
    print("Metrics:", metrics)

    plot_spring_results(t_test.ravel(), u_true.ravel(), u_pred.ravel(),
                        t_obs.ravel(), u_obs.ravel(), history, C_TRUE, K_TRUE,
                        path=os.path.join(out_dir, "spring_inverse_results.png"),
                        show=args.show)


if __name__ == "__main__":
    main()
