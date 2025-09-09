import numpy as np
import fracnum as fr

from fracnum.splines import BernsteinSplines
from fracnum.plotting_utils import VdP_Plotter
from fracnum.numerical import build_hilf_knot_vals

"""
Example file of using Bernstein splines for time-integrating the (forced) fractional Van der Pol oscillator in the sense:

x'' - mu * (1 - x^2) * D^{alpha, \beta} x + x = A * sin(omega * t)

Where D represents the Hilfer derivative of order alpha and type beta.
Enjoy the stability, quasiperiodicity and chaos!

- Niels Goedegebure, March 5, 2025

"""

######################
# VdP function setup #
######################

alpha_damping = 0.8  # Fractional order of damping
beta = 1  # Hilfer Beta

params = {"mu": 1}  # mu parameter of VdP oscillator

# Forcing is a list of dictionaries applying A*sin(omega*t) + c to solution component dim (1 gives y in this case)
forcing_params = [{"dim": 1, "A": 3, "omega": 3.3, "c": 0}]  # 3  # 1.94,#4.0,#6.2,

# Initialize the function for the Van der Pol oscillator
# This has to be done before the next steps in order to get the right N_upscale. See below.
VdP_bs = fr.ode_functions.VdP(params, bernstein=True, transpose=False)

###################
# Spline settings #
###################

### Knot input values ###
T = 100  # Integration max time
eps = 10 ** (-15)  # Time shift epsilon, start of interval
dt = 0.05  # Spline size (also called h) though varying size can also be used by creating a custom t_knot_vals
c = 3 / 2  # Knot size increase constant
##

gamm = alpha_damping + beta - alpha_damping * beta  # Hilfer gamma parametrization
t_knot_vals = build_hilf_knot_vals(
    eps, T, c, gamm, dt
)  # Knot values: equidistant for Caputo (beta = 1)

####

n_eval = 1  # Polynomial order

# A bit of a technicality: the order of the polynomial goes up by 3 since it gets multiplied 3 times in the VdP system equation.
# In general equal to m * n_eval, where m denotes the highest order of multiplication in x-components of f(x)
n_calc = VdP_bs.N_upscale * n_eval

bs = BernsteinSplines(
    t_knot_vals, n_calc, n_eval=n_eval, eq_opt=True
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
x_0 = np.array([0, 0])

# alpha for x and y component separately. For reference, see https://doi.org/10.1016/j.chaos.2006.05.010
system_alpha_vals = [alpha_damping, 2 - alpha_damping]
# Initialize the solving structure
spline_solver = bs.initialize_solver(
    VdP_bs.f, x_0, system_alpha_vals, beta_vals=beta, forcing_params=forcing_params
)
# Run the solver. method = 'local' uses knot-by-knot integration, method = 'global' at-large interval integration.
results = spline_solver.run(method="local", verbose=False, t_eval=t_knot_vals)

# Save some results...
x = results["x"][0]  # Function at evaluation points as vector
a_vals = results["a"]  # Splines coefficients in matrix form
comp_time = results["total_time"]  # Total computational time
t = results["t"]  # Knot time values

x_der = np.diff(x) / np.diff(t)  # Calculate the derivative of x

############
# Plotting #
############

skip_n_vals = 20  # Skip first n vals in case of Hilfer derivative for plotting
if np.all(beta == 1):
    skip_n_vals = 0

plot_x = x[skip_n_vals:-1]  # Last value is discarded for the derivative calculation
plot_x_der = x_der[skip_n_vals:]
plot_t = t[skip_n_vals:-1]

# If provided, can be used to keep fixed limits for the frame. Currently not inputted below.
lims_override = {"x": [-2.0, 2.0], "xder": [-5.0, 5.0], "fourier_amp": [0, 2.8]}

plot_object = VdP_Plotter(
    plot_x,
    plot_x_der,
    plot_t,
    params,
    alpha_damping,
    dt,
    T,
    n_eval,
    comp_time,
    forcing_params=forcing_params,
    lims_override=None,
    beta=beta,
)

plot_object.phase()
plot_object.phase(empty=True)
plot_object.threedee()
plot_object.fourier_spectrum()
plot_object.phase_fourier()

plot_object.show_plots()

############
# Enjoy!:) #
############
