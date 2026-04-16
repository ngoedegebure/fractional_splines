# %%
import numpy as np
import fracnum as fr
import matplotlib.pyplot as plt
from fracnum.plotting_utils import get_lin_line_colors
from scipy.special import gamma, betainc

def print_sc(output_str, remove_zero_exp = False):
    output_str_format = output_str.replace("e+0", " \cdot 10^{").replace("e-0", " \cdot 10^{-")
    
    if remove_zero_exp:
        print(output_str_format.replace(r"\cdot 10^{0 }", ""))
    else:
        print(output_str_format)

# %%

SAVE_TO_PDF = False
PLOT_SELECTION = ["pol"]

# %%
### Initialize rhs function f ###

# D^\alpha = t**k + c

y_0 = np.array([1])

beta = 0.5
eps = 1e-10

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

# %%
### Simulation parameters ###

CONV_TOL = 1e-12

# %%
### h plot parameters ###

# %%
alpha_vals = np.linspace(0.1, 1, 10)

if "pol" in PLOT_SELECTION:
    T = 4
    q = 2
    h = 2 ** (-1)
    detail_i_select = [0, 4, -1]
    alpha_i_select = [alpha_vals[i] for i in detail_i_select]

    ### h plot execution ###

    t_hr_eval = (
        np.linspace(eps / T, 1, int((1e3)))
    ) * T  # high-res time values
    colors, cmap = get_lin_line_colors(alpha_vals)
    mean_error_s, error_s = np.zeros(len(alpha_vals)), np.zeros(len(alpha_vals))
    run_times = np.zeros([len(alpha_vals)])
    spline_its = np.zeros(len(alpha_vals))

    fig, axs = plt.subplots(1, 3, figsize=(3 * fig_size, fig_size), layout="tight")

    i = 0
    for alpha in alpha_vals:

        gamm = alpha + beta - alpha * beta

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

        label_str = rf"$\alpha = {alpha:.1f}$"

        run_times[i] = np.array([run_time_s])
        spline_its[i] = res["n_it_per_knot"]
        axs[0].plot(t, y_q_eps, label=label_str, color=colors[i])

        error_time_weighed = t ** (1 - gamm) * (y_q_eps - y_eps_vals)

        if alpha in alpha_i_select:
            error = (y_q_eps - y_eps_vals)
            axs[2].plot(t, (np.abs(error)), label = rf'$\alpha = {alpha:.1f}$')

        mean_error_s[i] = np.mean(np.abs(error_time_weighed))
        error_s[i] = np.max(np.abs(error_time_weighed))

        linestyle = (0, (1,2))
        linewidth = 3
        if i == len(alpha_vals)-1:
            axs[0].plot(
                t, y_eps_vals, linestyle=linestyle, linewidth=linewidth, color=colors[i], label=r'Analytical $y^\varepsilon$'
            )
        else:
            axs[0].plot(
                t, y_eps_vals,linestyle=linestyle, linewidth=linewidth, color=colors[i]
            )

        i += 1

    axs[0].set_xlabel("$t$")
    axs[0].set_ylabel(r"$y\,(t)$")
    axs[0].set_title(rf"Solutions for $\alpha$", fontsize=small_font)
    axs[0].legend()
    axs[0].set_ylim([0, 10])

    order = alpha
    order_label = r"Theoretical upper bound $\mathcal{O}(h^\alpha)$"
    C = error_s[0] / (alpha_vals[0] ** order)
    reference_errors = C * alpha_vals**order

    ### Convergence order plot loglog ###
    axs[1].set_title(r"Absolute weighted error for $\alpha$", fontsize=small_font)
    axs[1].set_xlabel(r"$\alpha$")
    axs[1].set_ylabel(
        r"Weighted sup error $||y^{q,\varepsilon} - y^\varepsilon||_{1-\gamma}$ (log)"
    )
    axs[1].plot(alpha_vals, error_s, label="Numerical error", linewidth=2, color="orange")
    axs[1].legend()
    axs[1].invert_xaxis()

    ### Detailed view (i = detail_i_select) ###
    axs[2].set_title(
        rf"Absolute unweighted error over time",
        fontsize=small_font,
    )
    axs[2].set_xlabel("$t$")
    axs[2].set_ylabel(
        r"Unweighted error $|y^{q,\varepsilon}(t) - y^\varepsilon (t)|$ "
    )
    axs[2].legend()

    if SAVE_TO_PDF:
        plt.savefig(
            f'figures/conv_h--alpha_{str(alpha).replace(".", "_")}_k_{str(k).replace(".", "_")}.pdf',
            bbox_inches="tight",
        )
    else:
        plt.show()

    print("\n~~~h TABLE ~~~\n")
    print(r"$\alpha$ & mean weighted error & sup weighted error & total time (s) \\ \hline")
    for i_h in range(len(alpha_vals)):
        print_sc(fr"${alpha:.1f}^{{-{i_h}}}$ &$ {mean_error_s[i_h]:.3e} }}$& ${error_s[i_h]:.3e}}}$ & ${run_times[i_h]:.3e}}}$ \\")
    print(r'\hline'+"\n")
