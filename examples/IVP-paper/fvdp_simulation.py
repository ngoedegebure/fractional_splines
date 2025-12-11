import numpy as np
import fracnum as fr
import matplotlib.pyplot as plt

from fracnum.splines import BernsteinSplines
from fracnum.plotting_utils import get_lin_line_colors
from fracnum.ode_functions import ODEFunctions
from fracnum.numerical import build_hilf_knot_vals

# %%
SAVE_TO_PDF = True


# %%
class VdP_4(ODEFunctions):
    def __init__(self, params={}, bernstein=False, transpose=False):
        super().__init__(params, bernstein, transpose)
        self.N_upscale = 3

    def f(self, t_vals, x_vals):
        mu = self.params["mu"]

        x, y, z, u = x_vals[0, :], x_vals[1, :], x_vals[2, :], x_vals[3, :]

        # x' = y
        x_out = self.bs_upscale(y, self.N_upscale)

        # y' = z
        y_out = self.bs_upscale(z, self.N_upscale)

        # z' = u
        z_out = self.bs_upscale(u, self.N_upscale)

        # z' = \mu (1-x(t)^2)y(t) - x(t)
        u_out = mu * self.bs_mult(1 - self.bs_mult(x, x), y) - self.bs_upscale(
            x, self.N_upscale
        )

        return np.array([x_out, y_out, z_out, u_out])


# %%
beta_vals = [0, 0.25, 0.5, 0.75, 1]
n_beta = len(beta_vals)
x_vals = [None] * n_beta
x_der_vals = [None] * n_beta
t_vals = [None] * n_beta
res_vals = [None] * n_beta
i = 0
for beta in beta_vals:
    alpha_damping = 0.5  # Fractional order of damping
    beta = beta_vals[i]

    gamm = alpha_damping + beta - alpha_damping * beta

    eps = 1e-5
    T = 100  # Integration max time
    dt = 0.05  # Spline size (also called h) though varying size can also be used by creating a custom t_knot_vals
    c = 1.5  # growth factor of t values

    t_knot_vals = build_hilf_knot_vals(eps, T, c, gamm, dt)

    ######################
    # VdP function setup #
    ######################

    params = {"mu": 1}  # mu parameter of VdP oscillator

    # Initialize the function for the Van der Pol oscillator
    # This has to be done before the next steps in order to get the right N_upscale. See below.
    VdP_bs = VdP_4(params, bernstein=True, transpose=False)

    ###################
    # Spline settings #
    ###################

    q_eval = 1  # Polynomial order. NOTE: when taking the spline derivative as done below, it is beneficial to keep this at 1 for plotting / spline cont. diff reasons

    ####
    # A bit of a technicality: the order of the polynomial goes up by 3 since it gets multiplied 3 times in the VdP system equation.
    # In general equal to m * q_eval, where m denotes the highest order of multiplication in x-components of f(x)
    q_calc = VdP_bs.N_upscale * q_eval

    bs = BernsteinSplines(
        t_knot_vals, q_calc, n_eval=q_eval
    )  # Initialize splines setup!

    mult = bs.splines_multiply  # Bernstein multiplication method
    upscale = (
        bs.splines_upscale
    )  # Bernstein upscale method to match the polynomial order (same as multiplying n times with identity splines of the same order)

    VdP_bs.set_bs_mult_upscale_functions(mult, upscale)

    ############
    # Compute! #
    ############

    # Initial value (x, y).
    # NOTE: for now, keep y to zero so that the derivative calculation still applies without need for a constant
    x_0 = np.array([1, 0, 0, 0])

    dt_eval = 0.01
    t_eval = t_knot_vals  # np.linspace(eps, T, int((T-eps)/dt_eval)+1)

    # Initialize the solving structure
    spline_solver = bs.initialize_solver(VdP_bs.f, x_0, alpha_damping, beta_vals=beta)
    # Run the solver. method = 'local' uses knot-by-knot integration, method = 'global' at-large interval integration.
    results = spline_solver.run(
        t_eval=t_eval, method="local", verbose=False, conv_tol=1e-12, conv_max_it=5000
    )

    # Save some results...
    # breakpoint()
    x = results["x"][0, :-1]
    comp_time = results["total_time"]  # Total computational time
    t = results["t"]  # Knot time values
    x_der = np.diff(x) / np.diff(t_eval[:-1])

    t_vals[i] = t_eval

    x_vals[i] = x
    x_der_vals[i] = x_der

    res_vals[i] = results

    i += 1
# %%
colors, cmap = get_lin_line_colors(beta_vals)

aspect = 2 / 3
size = 5
fig = plt.figure(figsize=(size, size))

for i in range(n_beta):
    x, x_der = x_vals[i], x_der_vals[i]
    plt.plot(x[:-1], x_der, label=f"{beta_vals[i]:.2f}", color=colors[i])

margin = 1
max_x = 2.25 * margin
xlims = np.array([-max_x, max_x])
plt.xlim(xlims)

ylims = 1 / aspect * xlims
plt.ylim(ylims)
plt.xlim(ylims)
plt.xlabel("$x$"), plt.ylabel("$\dot{x}$")
plt.legend(title=r"$\beta$")

if SAVE_TO_PDF:
    plt.savefig("figures/VdP_phase.pdf", bbox_inches="tight")
else:
    plt.show()

# %%
aspect = 1 / 3
size = 5
fig, ax = plt.subplots(2, 1, figsize=(size, aspect * size * 2))

i_beta_plot_select = range(n_beta)  # [0, -1]

T_time_plot = 15
for i in i_beta_plot_select:
    gamm = alpha_damping + beta_vals[i] - alpha_damping * beta_vals[i]
    t = build_hilf_knot_vals(eps, T_time_plot, c, gamm, dt)[:-1]
    x = x_vals[i]
    x_der = x_der_vals[i]
    ax[0].plot(t, x[: len(t)], label=f"{beta_vals[i]:.2f}", color=colors[i])
    ax[1].plot(
        t[:-1], x_der[: len(t) - 1], label=f"{beta_vals[i]:.2f}", color=colors[i]
    )

plt.xlabel("$t$"), ax[0].set_ylabel("$x$"), ax[1].set_ylabel(r"$\dot{x}$")
ax[0].set_ylim(ylims), ax[1].set_ylim(ylims)
ax[0].set_xticklabels([])
if SAVE_TO_PDF:
    plt.savefig("figures/VdP_time.pdf", bbox_inches="tight")
else:
    plt.show()
# %%

print("\n")
print(
    r"$\beta$ & knots & $x(\eps)$ & total time (s) &  it. per knot & time per it. (s) \\ \hline"
)

for i in range(n_beta):
    gamm = alpha_damping + beta_vals[i] - alpha_damping * beta_vals[i]
    t = build_hilf_knot_vals(eps, T, c, gamm, dt)[:-1]
    N = len(t)
    output_str = (
        rf"${beta_vals[i]:.2f}$ & {N} & ${x_vals[i][0]:.3e} $& ${res_vals[i]['total_time']:.3f}$ & ${res_vals[i]['n_it_per_knot']:.3f}$&$ {res_vals[i]['time_per_it']:.3e} "
        + "}$"
        + rf"\\ "
    )
    print(output_str.replace("e+0", " \cdot 10^").replace("e-0", " \cdot 10^{-"))
print("\n")
