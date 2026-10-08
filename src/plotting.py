"""
Figures in the same style as the paper (Fig. 2, 3 and 4):
reference surface | PINN surface | absolute error map | loss curves
"""
import numpy as np


def plot_field_results(Xg, Yg, ref, pred, history, curves, xlabel, ylabel,
                       zlabel, ref_title, path, show=False, err_mask=None,
                       err_title="Absolute error"):
    """
    err_mask : optional boolean array, True where the error should NOT be
               drawn (e.g. the t = 0 row of the consolidation problem, where
               the exact solution has a jump).
    """
    import matplotlib.pyplot as plt

    err = np.abs(pred - ref)
    if err_mask is not None:
        err = np.ma.masked_where(err_mask, err)
    vmin, vmax = ref.min(), ref.max()
    fig = plt.figure(figsize=(13, 10))

    for i, (Z, title) in enumerate([(ref, ref_title), (pred, "PINN solution")]):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        surf = ax.plot_surface(Xg, Yg, Z, cmap="jet", vmin=vmin, vmax=vmax,
                               linewidth=0, antialiased=True)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        fig.colorbar(surf, ax=ax, shrink=0.6)

    ax = fig.add_subplot(2, 2, 3)
    pc = ax.pcolormesh(Xg, Yg, err, cmap="jet", shading="auto")
    fig.colorbar(pc, ax=ax, label="|error|")
    ax.set_title(err_title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax = fig.add_subplot(2, 2, 4)
    for key, label in curves:
        if key in history:
            ax.semilogy(history["iter"], history[key], label=label)
    ax.set_xlabel("# Iterations")
    ax.set_ylabel("Loss")
    ax.set_title("Convergence history")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved figure: {path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_spring_results(t_test, u_true, u_pred, t_obs, u_obs, history,
                        c_true, k_true, path, show=False):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    ax = axes[0]
    ax.plot(t_obs, u_obs, "o", ms=4, mfc="none", label="Observations")
    ax.plot(t_test, u_true, "k-", lw=1.5, label="True solution")
    ax.plot(t_test, u_pred, "r-.", lw=1.5, label="PINN prediction")
    ax.set_xlabel("t")
    ax.set_ylabel("u")
    ax.set_title("Displacement")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(history["iter"], history["k"], "r-", label="Identified k")
    ax.axhline(k_true, color="r", ls="-.", label="True k")
    ax.plot(history["iter"], history["c"], "b-", label="Identified c")
    ax.axhline(c_true, color="b", ls="-.", label="True c")
    ax.set_xlabel("# Iterations")
    ax.set_ylabel("Parameter value")
    ax.set_title("Parameter identification")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[2]
    for key, label in [("L_f", "ODE train loss"), ("test_pde", "ODE test loss"),
                       ("L_d", "Observations MSE")]:
        if key in history:
            ax.semilogy(history["iter"], history[key], label=label)
    ax.set_xlabel("# Iterations")
    ax.set_ylabel("Loss")
    ax.set_title("Convergence history")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved figure: {path}")
    if show:
        plt.show()
    plt.close(fig)
