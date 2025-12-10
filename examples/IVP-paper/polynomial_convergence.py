# %%
import numpy as np
import fracnum as fr
import matplotlib.pyplot as plt
from fracnum.plotting_utils import get_lin_line_colors
from scipy.special import gamma, betainc
import time
from fracnum.splines import BernsteinMethods

# %%

# PLOT_SELECTION = ["h", "q", "eps"]
PLOT_SELECTION = ["h", "q", "eps"]

# %%
### Initialize rhs function f ###

# D^\alpha = t**k + c

y_0 = np.array([1])
alpha = 0.5
beta = 0.5
eps = 1e-10
gamm = alpha + beta - alpha * beta

k = 0.9
params = {"k": np.array([k]), "c": 0}
f = fr.ode_functions.t_k(params=params, bernstein=True, transpose=False)

### Define solution function y and shifted solution y_eps ###


def y(t, k, x_0, alpha, beta):
    gamm = alpha + beta - alpha * beta
    return x_0 * t ** (gamm - 1) / gamma(gamm) + gamma(k + 1) / gamma(
        alpha + k + 1
    ) * t ** (alpha + k)


def y_eps(t, k, x_0, alpha, beta, eps):
    gamm = alpha + beta - alpha * beta
    betainc_part = (
        betainc(alpha, k + 1, 1 - eps / t)
        * gamma(alpha)
        * gamma(k + 1)
        / gamma(alpha + k + 1)
    )
    return (
        x_0 * t ** (gamm - 1) / gamma(gamm)
        + t ** (alpha + k) / gamma(alpha) * betainc_part
    )


# %%
### Figure parameters ###
fig_size = 4.5
big_font, small_font = 12.5, 11.5
SAVE_TO_PDF = True

# %%
### Simulation parameters ###

T = 4
q = 1
CONV_TOL = 1e-12

# %%
### h plot parameters ###

base_h = 1 / 2
h_vals = base_h ** np.arange(0, 9)
detail_i_select = 4

# %%

if "h" in PLOT_SELECTION:

    ### h plot execution ###

    t_hr_eval = (
        np.linspace(eps / T, 1, int(T / (base_h**10)))
    ) * T  # high-res time values
    colors, cmap = get_lin_line_colors(h_vals)
    error_s = np.zeros(len(h_vals))
    run_times = np.zeros([len(h_vals), 2])
    spline_its = np.zeros(len(h_vals))

    fig, axs = plt.subplots(1, 3, figsize=(3 * fig_size, fig_size), layout="tight")

    i = 0
    for h in h_vals:
        N = int(T / h) + 1
        t_knot_vals = (np.linspace(eps / T, 1, N)) ** 1 * T

        bs = fr.splines.BernsteinSplines(
            t_knot_vals, f.N_upscale * q, silent_mode=True, n_eval=q
        )
        f.bs_mult, f.bs_upscale = bs.splines_multiply, bs.splines_upscale
        solver = bs.initialize_solver(f.f, y_0, alpha, beta_vals=beta)

        res = solver.run(
            t_eval=t_hr_eval,
            verbose=False,
            conv_tol=CONV_TOL,
            method="local",
            conv_max_it=5000,
        )
        y_q_eps, t, run_time_s = np.squeeze(res["x"]), t_hr_eval, res["total_time"]
        y_eps_vals = y_eps(t, k, y_0, alpha, beta=beta, eps=eps)

        label_str = rf"$h\,={1/base_h:.0f}^{{-{i}}}$"

        run_times[i] = np.array([run_time_s])
        spline_its[i] = res["n_it_per_knot"]
        axs[0].plot(t, y_q_eps, label=label_str, color=colors[i])

        error_time_weighed = t ** (1 - gamm) * (y_q_eps - y_eps_vals)

        if i == detail_i_select:
            error = error_time_weighed

        error_s[i] = np.max(np.abs(error_time_weighed))

        i += 1

    ### Solution plot ###
    axs[0].plot(
        t, y_eps_vals, "--", label=r"Analytical $y^{\varepsilon}$", linewidth=2.5
    )
    axs[0].set_xlabel("$t$")
    axs[0].set_ylabel(r"$y\,(t)$")
    axs[0].set_title(rf"Solutions for knot size $h$", fontsize=small_font)
    axs[0].legend()
    axs[0].set_ylim([0, 10])

    order = alpha
    order_label = r"Theoretical upper bound $\mathcal{O}(h^\alpha)$"
    C = error_s[0] / (h_vals[0] ** order)
    reference_errors = C * h_vals**order

    ### Convergence order plot loglog ###
    axs[1].set_title(f"Convergence order", fontsize=small_font)
    axs[1].set_xlabel("Knot size $h$ (log, decreasing)")
    axs[1].set_ylabel(
        r"Weighed sup error $||y^{q,\varepsilon} - y^\varepsilon||_{1-\gamma}$ (log)"
    )
    axs[1].loglog(h_vals, error_s, label="Numerical error", linewidth=2, color="orange")
    axs[1].loglog(
        h_vals,
        reference_errors,
        "--",
        label=order_label,
        linewidth=2,
        color="lightseagreen",
    )
    axs[1].legend()
    axs[1].invert_xaxis()

    ### Detailed view (i = detail_i_select) ###
    axs[2].set_title(
        rf"Absolute weighed error over time $h\,={1/base_h:.0f}^{{-{detail_i_select}}}$",
        fontsize=small_font,
    )
    axs[2].plot(t, (np.abs(error)))
    axs[2].set_xlabel("$t$")
    axs[2].set_ylabel(
        r"Weighed error $|y^{q,\varepsilon}(t) - y^\varepsilon (t)|_{1-\gamma}$ "
    )

    if SAVE_TO_PDF:
        plt.savefig(
            f'figures/conv_h--alpha_{str(alpha).replace(".", "_")}_k_{str(k).replace(".", "_")}.pdf',
            bbox_inches="tight",
        )
    else:
        plt.show()

# %%
if "q" in PLOT_SELECTION:
    h = 2 ** (-1)
    base_q = 1 / 2
    q_vals = np.array(base_q ** (-np.arange(0, 5)), dtype="int")
    detail_q_select = 4

    colors, cmap = get_lin_line_colors(q_vals)
    error_q = np.zeros(len(q_vals))
    run_times = np.zeros([len(q_vals), 2])
    spline_its = np.zeros(len(q_vals))
    fig, axs = plt.subplots(1, 3, figsize=(3 * fig_size, fig_size), layout="tight")

    t_hr_eval = (np.linspace(eps / T, 1, int(T / (base_q**10)))) ** 1 * T

    i = 0
    for q in q_vals:
        N = int(T / h) + 1
        t_knot_vals = (np.linspace(eps / T, 1, N)) ** 1 * T

        bs = fr.splines.BernsteinSplines(
            t_knot_vals, f.N_upscale * q, silent_mode=True, n_eval=q
        )
        f.bs_mult, f.bs_upscale = bs.splines_multiply, bs.splines_upscale
        solver = bs.initialize_solver(f.f, y_0, alpha, beta_vals=beta)
        res = solver.run(
            t_eval=t_hr_eval,
            verbose=False,
            conv_tol=CONV_TOL,
            method="local",
            conv_max_it=5000,
        )

        label_str = f"$q\,={q}$"
        y_q_eps, t, run_time_s = np.squeeze(res["x"]), t_hr_eval, res["total_time"]
        y_eps_vals = y_eps(t, k, y_0, alpha, beta=beta, eps=eps)

        run_times[i] = np.array([run_time_s])
        spline_its[i] = res["n_it_per_knot"]

        axs[0].plot(t, y_q_eps, label=label_str, color=colors[i])

        error_time_weighed = t ** (1 - gamm) * (y_q_eps - y_eps_vals)

        if q == detail_q_select:
            error = error_time_weighed

        error_q[i] = np.max(np.abs(error_time_weighed))

        i += 1

    axs[0].plot(t, y_q_eps, "--", label=r"Analytical $y^{\varepsilon}$", linewidth=2.5)
    axs[0].set_xlabel("$t$")
    axs[0].set_ylabel("$y\,(t)$")
    axs[0].set_title(rf"Solutions for polynomial spline order $q$", fontsize=small_font)
    axs[0].legend()
    axs[0].set_ylim([0, 10])

    order = -alpha / 2
    order_label = r"Theoretical upper bound $\mathcal{O}(q^{-\alpha/2})$"
    C = error_q[0] / (q_vals[0] ** order)  # Compute the constant C
    reference_errors = C * q_vals**order

    axs[1].set_title(f"Convergence order", fontsize=small_font)
    axs[1].set_xlabel("Spline order $q$ (log)")
    axs[1].set_ylabel(
        r"Weighed sup error $||y^{q,\varepsilon} - y^\varepsilon||_{1-\gamma}$ (log)"
    )
    axs[1].loglog(q_vals, error_q, label="Numerical error", linewidth=2, color="orange")
    axs[1].loglog(
        q_vals,
        reference_errors,
        "--",
        label=order_label,
        linewidth=2,
        color="lightseagreen",
    )
    axs[1].legend()

    axs[2].set_title(
        f"Absolute weighed error over time $q={{{detail_q_select}}}$",
        fontsize=small_font,
    )
    axs[2].plot(t, np.abs(error))
    axs[2].set_xlabel("$t$")
    axs[2].set_ylabel(
        r"Weighed error $|y^{q,\varepsilon}(t) - y^\varepsilon (t)|_{1-\gamma}$ "
    )

    if SAVE_TO_PDF:
        plt.savefig(
            f'figures/conv_q--alpha_{str(alpha).replace(".", "_")}_k_{str(k).replace(".", "_")}.pdf',
            bbox_inches="tight",
        )
    else:
        plt.show()

# %%
if "eps" in PLOT_SELECTION:
    h = 2 ** (-1)
    q = 2
    base_eps = 1 / 2

    i_vals_eps = np.arange(1, 10)
    eps_vals = np.array(base_eps ** (i_vals_eps))

    detail_eps_i_select = [0, 1, 2, 3]

    colors, cmap = get_lin_line_colors(eps_vals)
    error_eps = np.zeros(len(eps_vals))

    run_times = np.zeros([len(eps_vals), 2])
    spline_its = np.zeros(len(eps_vals))

    fig, axs = plt.subplots(1, 3, figsize=(3 * fig_size, fig_size), layout="tight")

    i = 0
    for eps in eps_vals:
        t_hr_eval = (np.linspace(eps / T, 1, int(T / ((1 / 2) ** 9)))) * T
        N = int(T / h) + 1
        t_knot_vals = (np.linspace(eps / T, 1, N)) * T

        bs = fr.splines.BernsteinSplines(
            t_knot_vals, f.N_upscale * q, silent_mode=True, n_eval=q
        )
        f.bs_mult, f.bs_upscale = bs.splines_multiply, bs.splines_upscale
        solver = bs.initialize_solver(f.f, y_0, alpha, beta_vals=beta)
        res = solver.run(
            t_eval=t_hr_eval,
            verbose=False,
            conv_tol=CONV_TOL,
            method="local",
            conv_max_it=5000,
        )

        label_str = rf"$\varepsilon \,={1/base_eps:.0f}^{{-{i_vals_eps[i]}}}$"

        y_q_eps, t, run_time_s = np.squeeze(res["x"]), t_hr_eval, res["total_time"]
        y_vals = y(t, k, y_0, alpha, beta=beta)

        run_times[i] = np.array([run_time_s])
        spline_its[i] = res["n_it_per_knot"]

        axs[0].plot(t, y_q_eps, label=label_str, color=colors[i])

        error_time_weighed = t ** (1 - gamm) * (y_q_eps - y_vals)
        error_eps[i] = np.max(np.abs(error_time_weighed))

        if eps in eps_vals[detail_eps_i_select]:
            error = error_time_weighed
            axs[2].plot(
                t,
                np.abs(error),
                label=rf"$\varepsilon \,={1/base_eps:.0f}^{{-{i_vals_eps[i]}}}$",
            )

        i += 1

    axs[0].plot(t, y_vals, "--", label=r"Analytical $y$", linewidth=2.5)
    axs[0].set_xlabel("$t$")
    axs[0].set_ylabel(r"$y\,(t)$")
    axs[0].set_title(
        rf"Solutions for time domain shift $\varepsilon$", fontsize=small_font
    )
    axs[0].legend()
    axs[0].set_ylim([0, 10])

    order = alpha
    order_label = r"Theoretical upper bound $\mathcal{O}(\varepsilon^{\alpha})$"
    C = error_eps[0] / (eps_vals[0] ** order)
    reference_errors = C * eps_vals**order

    axs[1].set_title(f"Convergence order", fontsize=small_font)
    axs[1].set_xlabel(r"$\varepsilon$ (log, decreasing)")
    axs[1].set_ylabel(
        r"Weighed sup error $||y^{q,\varepsilon} - y\, ||_{1-\gamma}$ (log)"
    )
    axs[1].invert_xaxis()
    axs[1].loglog(
        eps_vals, error_eps, label="Numerical error", linewidth=2, color="orange"
    )
    axs[1].loglog(
        eps_vals,
        reference_errors,
        "--",
        label=order_label,
        linewidth=2,
        color="lightseagreen",
    )
    axs[1].legend()

    axs[2].set_title("Absolute weighed error over time", fontsize=small_font)
    axs[2].legend()
    axs[2].set_xlabel("$t$")
    axs[2].set_ylabel(r"Weighed error $|y^{q,\varepsilon}(t) - y (t)\,|_{1-\gamma}$ ")
    if SAVE_TO_PDF:
        plt.savefig(
            f'figures/conv_eps--alpha_{str(alpha).replace(".", "_")}_k_{str(k).replace(".", "_")}.pdf',
            bbox_inches="tight",
        )
    else:
        plt.show()
