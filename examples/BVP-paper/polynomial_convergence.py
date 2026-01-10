# %%
import numpy as np
import fracnum as fr
import matplotlib.pyplot as plt
from fracnum.plotting_utils import get_lin_line_colors
from scipy.special import gamma, betainc
import time
from fracnum.splines import BernsteinMethods

def print_sc(output_str, remove_zero_exp = False):
    output_str_format = output_str.replace("e+0", " \cdot 10^{").replace("e-0", " \cdot 10^{-")
    
    if remove_zero_exp:
        print(output_str_format.replace(r"\cdot 10^{0 }", ""))
    else:
        print(output_str_format)

# %%

PLOT_SELECTION = ["h","q","eps"]
SAVE_TO_PDF = False

# %%
### Initialize rhs function f ###

# D^\alpha = t**k + c

y_0 = np.array([1])
alpha = 0.5
beta = 0.5
eps = 1e-10
gamm = alpha + beta - alpha * beta

k = 0.9
c = 0

params = {'k':np.array([k]), 'c':c}
f = fr.ode_functions.t_k(params = params, bernstein=True,  transpose = False)

### Define solution function y and shifted solution y_eps ###


def y(t, T, k,x_0, alpha, beta, eps=0, bvp = False):
    gamm = alpha + beta - alpha*beta
    zeta = 1-gamm+alpha

    ivp_part = x_0*t**(gamm-1)/gamma(gamm)+ gamma(k+1)/gamma(alpha+k+1)*t**(alpha+k) #x_0*t**(gamm-1)/gamma(gamm)+t**(alpha+k)*gamma(k+1)/gamma(alpha+k+1)
    if bvp:
        prefix = t**alpha/gamma(alpha+1)
    else:
        prefix = 0
    delta = - gamma(zeta+1)/T**zeta * T**(zeta+k)*gamma(k+1)/gamma(zeta+k+1)
    print(f"delta an: {delta}")
    return ivp_part +prefix * delta


def y_eps(t, T, k,x_0, alpha, beta, eps=0, bvp=False):
    gamm = alpha + beta - alpha*beta
    zeta = alpha + 1-gamm
    
    betainc_part_t = betainc(alpha, k+1, 1-eps/t) * gamma(alpha)*gamma(k+1)/gamma(alpha+k+1)
    betainc_part_T = betainc(zeta, k+1, 1-eps/T) * gamma(zeta)*gamma(k+1)/gamma(zeta+k+1)
    ivp_part = x_0*t**(gamm-1)/gamma(gamm)
    if bvp:
        return ivp_part + t**(alpha+k)/gamma(alpha)* betainc_part_t\
              -(gamma(zeta+1)*t**alpha)/(gamma(alpha+1)*T**zeta)*T**(zeta+k)/gamma(zeta)*betainc_part_T
    else:
        return ivp_part


# %%
### Figure parameters ###
fig_size = 4.5
big_font, small_font = 12.5, 11.5
plot_0_ylims = [-1,5]

# %%
### Simulation parameters ###

T = 3
T_bvp = T
q = 1
CONV_TOL = 1e-12
BVP = True
HR_RES = 7500
CONV_MAX_IT = 5000

# %%
### h plot parameters ###

base_h = 1 / 2
h_vals = base_h ** np.arange(0, 9)
detail_i_select =  [0, 1, 2, 3, len(h_vals)-1]

# %%

if "h" in PLOT_SELECTION:

    ### h plot execution ###

    t_hr_eval = (np.linspace(eps/T,1, HR_RES))**1*T # high-res time values
    colors, cmap = get_lin_line_colors(h_vals)
    mean_error_s, error_s = np.zeros(len(h_vals)), np.zeros(len(h_vals))
    delta_s, delta_an = np.zeros(len(h_vals)), np.zeros(len(h_vals))
    run_times = np.zeros([len(h_vals)])
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
            method="global",
            bvp = BVP,
            T = T_bvp,
            save_x=False,
            conv_max_it=CONV_MAX_IT,
        )
        y_q_eps, t, run_time_s, delta = np.squeeze(res["x"]), t_hr_eval, res["total_time"], np.array(res['delta'])
        y_eps_vals = y_eps(t, T_bvp, k,y_0, alpha, beta, eps=0, bvp = BVP)

        label_str = rf"$h\,={1/base_h:.0f}^{{-{i}}}$"

        run_times[i] = np.squeeze([run_time_s])
        spline_its[i] = res["n_it_per_knot"]
        axs[0].plot(t, y_q_eps, label=label_str, color=colors[i])

        error_time_weighed = t ** (1 - gamm) * (y_q_eps - y_eps_vals)

        if i in detail_i_select:
            error = error_time_weighed
            axs[2].plot(t, (np.abs(error)), label = label_str, color=colors[i])

        mean_error_s[i] = np.mean(np.abs(error_time_weighed))
        error_s[i] = np.max(np.abs(error_time_weighed))
        delta_s[i] = delta

        i += 1

    ### Solution plot ###
    axs[0].plot(
        t, y_eps_vals, "--", label=r"Analytical $x^{\varepsilon}$", linewidth=2.5
    )
    axs[0].set_xlabel("$t$")
    axs[0].set_ylabel(r"$x\,(t)$")
    axs[0].set_title(rf"Solutions for knot size $h$", fontsize=small_font)
    axs[0].legend()
    axs[0].set_ylim(plot_0_ylims)

    order = alpha
    order_label = r"Theoretical upper bound $\mathcal{O}(h^\alpha)$"
    C = error_s[0] / (h_vals[0] ** order)
    reference_errors = C * h_vals**order

    ### Convergence order plot loglog ###
    axs[1].set_title(f"Convergence order", fontsize=small_font)
    axs[1].set_xlabel("Knot size $h$ (log, decreasing)")
    axs[1].set_ylabel(
        r"Weighed sup error $||x^{q,\varepsilon} - x^\varepsilon||_{1-\gamma}$ (log)"
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
        rf"Absolute weighed error over time",
        fontsize=small_font,
    )
    axs[2].set_xlabel("$t$")
    axs[2].set_ylabel(
        r"Weighed error $|x^{q,\varepsilon}(t) - x^\varepsilon (t)|_{1-\gamma}$ "
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
    print(r"$h$ & mean weighed error & sup weighed error & delta & total time (s) \\ \hline")
    for i_h in range(len(h_vals)):
        print_sc(fr"${1/base_h:.0f}^{{-{i_h}}}$ &$ {mean_error_s[i_h]:.3e} }}$& ${error_s[i_h]:.3e}}}$ & ${delta_s[i_h]:.3e}}}$ &  ${run_times[i_h]:.3e}}}$ \\")
    print(r'\hline'+"\n")

# %%
if "q" in PLOT_SELECTION:
    h = 1e-2
    base_q = 1 / 2
    q_vals = np.array(base_q ** (-np.arange(0, 5)), dtype="int")
    detail_q_select = q_vals

    colors, cmap = get_lin_line_colors(q_vals)
    error_q, mean_error_q = np.zeros(len(q_vals)), np.zeros(len(q_vals))
    delta_s, delta_an = np.zeros(len(q_vals)), np.zeros(len(q_vals))
    run_times_q = np.zeros([len(q_vals)])
    spline_its = np.zeros(len(q_vals))
    fig, axs = plt.subplots(1, 3, figsize=(3 * fig_size, fig_size), layout="tight")

    t_hr_eval = (np.linspace(eps/T,1, HR_RES))**1*T # high-res time values

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
            method="global",
            conv_max_it=CONV_MAX_IT,
            bvp=True, T = T_bvp, save_x=False
        )

        label_str = f"$q\,={q}$"
        y_q_eps, t, run_time_s, delta = np.squeeze(res["x"]), t_hr_eval, res["total_time"], np.squeeze(res['delta'])
        y_eps_vals = y_eps(t, T_bvp, k,y_0, alpha, beta, eps=0, bvp = BVP)

        delta_s[i] = delta

        run_times_q[i] = np.squeeze([run_time_s])
        spline_its[i] = res["n_it_per_knot"]

        axs[0].plot(t, y_q_eps, label=label_str, color=colors[i])

        error_time_weighed = t ** (1 - gamm) * (y_q_eps - y_eps_vals)

        if q in detail_q_select:
            error = error_time_weighed
            axs[2].plot(t, (np.abs(error)), label = label_str, color=colors[i])

        error_q[i] = np.max(np.abs(error_time_weighed))
        mean_error_q[i] = np.mean(np.abs(error_time_weighed))

        i += 1

    axs[0].plot(t, y_q_eps, "--", label=r"Analytical $x^{\varepsilon}$", linewidth=2.5)
    axs[0].set_xlabel("$t$")
    axs[0].set_ylabel("$x\,(t)$")
    axs[0].set_title(rf"Solutions for polynomial spline order $q$", fontsize=small_font)
    axs[0].legend()
    axs[0].set_ylim(plot_0_ylims)

    order = -alpha / 2
    order_label = r"Theoretical upper bound $\mathcal{O}(q^{-\alpha/2})$"
    C = error_q[0] / (q_vals[0] ** order)  # Compute the constant C
    reference_errors = C * q_vals**order

    axs[1].set_title(f"Convergence order", fontsize=small_font)
    axs[1].set_xlabel("Spline order $q$ (log)")
    axs[1].set_ylabel(
        r"Weighed sup error $||x^{q,\varepsilon} - x^\varepsilon||_{1-\gamma}$ (log)"
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

    axs[2].set_title(rf"Absolute weighed error over time (detail)", fontsize=small_font)
    axs[2].set_xlabel("$t$")
    axs[2].set_ylabel(
        r"Weighed error $|x^{q,\varepsilon}(t) - x^\varepsilon (t)|_{1-\gamma}$ "
    )
    axs[2].set_xlim(-0.008,0.15)
    axs[2].legend()

    if SAVE_TO_PDF:
        plt.savefig(
            f'figures/conv_q--alpha_{str(alpha).replace(".", "_")}_k_{str(k).replace(".", "_")}.pdf',
            bbox_inches="tight",
        )
    else:
        plt.show()

    print("\n~~~q TABLE ~~~\n")
    print(r"$q$ & mean weighed error & sup weighed error & $\Delta$ & total time (s) \\ \hline")
    for i_q in range(len(q_vals)):
        print_sc(fr"${q_vals[i_q]:.0f}$ &$ {mean_error_q[i_q]:.3e} }}$& ${error_q[i_q]:.3e}}}$  & ${delta_s[i_q]:.3e}}}$ & ${run_times_q[i_q]:.3e}}}$ \\")
    print(r'\hline'+"\n")

# %%
if "eps" in PLOT_SELECTION:
    h = 10 ** (-2)
    q = 1
    base_eps = 1 / 2

    i_vals_eps = np.arange(1, 10)
    eps_vals = np.array(base_eps ** (i_vals_eps))

    detail_eps_i_select =[0, 1, 2, 3, 4, len(eps_vals)-1]

    colors, cmap = get_lin_line_colors(eps_vals)
    error_eps, mean_error_eps = np.zeros(len(eps_vals)), np.zeros(len(eps_vals))

    run_times_eps = np.zeros(len(eps_vals))
    spline_its = np.zeros(len(eps_vals))
    first_val = np.zeros(len(eps_vals))
    delta_s, delta_an = np.zeros(len(eps_vals)), np.zeros(len(eps_vals))

    fig, axs = plt.subplots(1, 3, figsize=(3 * fig_size, fig_size), layout="tight")

    i = 0
    for eps in eps_vals:
        t_hr_eval = (np.linspace(eps/T,1, HR_RES))**1*T # high-res time values
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
            method="global",
            conv_max_it=CONV_MAX_IT,
            bvp=True, T = T_bvp, save_x=False
        )

        label_str = rf"$\varepsilon \,={1/base_eps:.0f}^{{-{i_vals_eps[i]}}}$"

        y_q_eps, t, run_time_s, delta = np.squeeze(res["x"]), t_hr_eval, res["total_time"], np.squeeze(res["delta"])
        y_vals = y(t, T_bvp, k,y_0, alpha, beta, eps=0, bvp = BVP)
        delta_s[i] = delta

        first_val[i] = y_q_eps[0]
        run_times_eps[i] = np.squeeze([run_time_s])
        spline_its[i] = res["n_it_per_knot"]

        axs[0].plot(t, y_q_eps, label=label_str, color=colors[i])

        error_time_weighed = t ** (1 - gamm) * (y_q_eps - y_vals)
        error_eps[i] = np.max(np.abs(error_time_weighed))
        mean_error_eps[i] = np.max(np.abs(error_time_weighed))

        if eps in eps_vals[detail_eps_i_select]:
            error = error_time_weighed
            axs[2].plot(
                t,
                np.abs(error),
                label=label_str,
                color = colors[i]
            )

        i += 1

    axs[0].plot(t, y_vals, "--", label=r"Analytical $x$", linewidth=2.5)
    axs[0].set_xlabel("$t$")
    axs[0].set_ylabel(r"$x\,(t)$")
    axs[0].set_title(
        rf"Solutions for time domain shift $\varepsilon$", fontsize=small_font
    )
    axs[0].legend()
    axs[0].set_ylim(plot_0_ylims)

    order = alpha
    order_label = r"Theoretical upper bound $\mathcal{O}(\varepsilon^{\alpha})$"
    C = error_eps[0] / (eps_vals[0] ** order)
    reference_errors = C * eps_vals**order

    axs[1].set_title(f"Convergence order", fontsize=small_font)
    axs[1].set_xlabel(r"$\varepsilon$ (log, decreasing)")
    axs[1].set_ylabel(
        r"Weighed sup error $||x^{q,\varepsilon} - x\, ||_{1-\gamma}$ (log)"
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
    axs[2].set_ylabel(r"Weighed error $|x^{q,\varepsilon}(t) - x (t)\,|_{1-\gamma}$ ")
    if SAVE_TO_PDF:
        plt.savefig(
            f'figures/conv_eps--alpha_{str(alpha).replace(".", "_")}_k_{str(k).replace(".", "_")}.pdf',
            bbox_inches="tight",
        )
    else:
        plt.show()
    
    print("\n~~~ eps TABLE ~~~\n")
    print(r"$\varepsilon$ & mean weighed error & sup weighed error & $\Delta $ & total time (s) & $x^{q,\varepsilon}(\varepsilon)$ \\ \hline")
    for i_eps in range(len(eps_vals)):
        print_sc(fr"${1/base_eps:.0f}^{{-{i_vals_eps[i_eps]}}}$ &$ {mean_error_eps[i_eps]:.3e} }}$& ${error_eps[i_eps]:.3e}}}$ & ${delta_s[i_h]:.3e}}}$ &  ${run_times_eps[i_eps]:.3e}}}$ & ${first_val[i_eps]:.3e}}}$ \\")
    print(r'\hline'+"\n")
