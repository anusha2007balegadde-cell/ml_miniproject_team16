"""
Generic training loop used by all three problems.

Paper's optimisation procedure (Section 3):
    1. Adam for a fixed number of epochs
    2. then (optionally) L-BFGS until convergence
"""
import time

import torch


def train(params, loss_fn, adam_epochs, lr, lbfgs_iters=0,
          log_every=500, eval_fn=None):
    """
    params      : list of trainable tensors (network weights, plus c and k
                  for the inverse problem)
    loss_fn     : function returning (total_loss_tensor, dict_of_floats)
    adam_epochs : number of Adam steps
    lr          : Adam learning rate
    lbfgs_iters : max L-BFGS iterations after Adam (0 = skip L-BFGS)
    log_every   : how often to print and record the losses
    eval_fn     : optional function returning a dict of test metrics
    returns     : history dict, used later for the loss plots
    """
    history = {"iter": [], "loss": [], "time_s": []}
    t0 = time.time()

    def log(it, loss_value, parts):
        history["iter"].append(it)
        history["loss"].append(loss_value)
        history["time_s"].append(time.time() - t0)
        for k, v in parts.items():
            history.setdefault(k, []).append(v)
        metrics = eval_fn() if eval_fn is not None else {}
        for k, v in metrics.items():
            history.setdefault(k, []).append(v)
        shown = {**parts, **metrics}
        msg = "  ".join(f"{k}={v:.3e}" for k, v in shown.items())
        print(f"[iter {it:>6d}] loss={loss_value:.3e}  {msg}", flush=True)

    # ---------------- Stage 1: Adam ----------------
    print(f"Adam: {adam_epochs} epochs, lr={lr}")
    optimizer = torch.optim.Adam(params, lr=lr)
    for it in range(1, adam_epochs + 1):
        optimizer.zero_grad()
        loss, parts = loss_fn()
        loss.backward()
        optimizer.step()
        if it == 1 or it % log_every == 0:
            log(it, loss.item(), parts)

    # ---------------- Stage 2: L-BFGS ----------------
    if lbfgs_iters > 0:
        print(f"L-BFGS: up to {lbfgs_iters} iterations")
        lbfgs = torch.optim.LBFGS(params, lr=1.0, max_iter=lbfgs_iters,
                                  max_eval=int(lbfgs_iters * 1.25),
                                  tolerance_grad=1e-9, tolerance_change=1e-12,
                                  history_size=50,
                                  line_search_fn="strong_wolfe")
        counter = {"n": 0}

        def closure():
            lbfgs.zero_grad()
            loss, parts = loss_fn()
            loss.backward()
            counter["n"] += 1
            if counter["n"] % log_every == 0:
                log(adam_epochs + counter["n"], loss.item(), parts)
            return loss

        lbfgs.step(closure)
        loss, parts = loss_fn()
        log(adam_epochs + counter["n"], loss.item(), parts)

    print(f"Training finished in {time.time() - t0:.1f} s")
    return history
