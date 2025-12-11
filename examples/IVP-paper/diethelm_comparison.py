import numpy as np
import time
import fracnum as fr
import matplotlib.pyplot as plt

from fracnum.splines import BernsteinSplines
from fracnum.plotting_utils import get_lin_line_colors
from fracnum.ode_functions import ODEFunctions
from fracnum.numerical import ivp_diethelm
from scipy.special import gamma as gamm

#%%
SAVE_AS_PDF = True
#%%

def print_sc(output_str, remove_zero_exp = False):
    output_str_format = output_str.replace("e+0", " \cdot 10^{").replace("e-0", " \cdot 10^{-")
    
    if remove_zero_exp:
        print(output_str_format.replace(r"\cdot 10^{0 }", ""))
    else:
        print(output_str_format)

def ML(z, alpha, beta=1.0, n_terms=50):
    z_arr = np.atleast_1d(z)
    
    # 1. Create the range of k values: [0, 1, ..., n_terms-1]
    k = np.arange(n_terms)
    
    # 2. Vectorized calculation of terms using broadcasting
    # z shape: (M,) -> (M, 1)
    # k shape: (N,)
    # Resulting 'terms' shape: (M, N)
    numerator = z_arr[:, np.newaxis] ** k
    denominator = gamm(alpha * k + beta)
    
    # 3. Sum along the terms axis (axis 1)
    result = np.sum(numerator / denominator, axis=1)
    
    # Return scalar if input was scalar
    if np.isscalar(z):
        return result[0]
    return result

#%%

# x' = a * x
class Lin_DE(ODEFunctions):
    def __init__(self, params={}, bernstein=False, transpose=False):
        super().__init__(params, bernstein, transpose)
        self.N_upscale = 1
        if 'a' in params.keys():
            self.a = params['a']
        else:
            self.a = 1

    def f(self, t_vals, x_vals):
        return self.a*x_vals

a = -1
params = {'a' : a}

f_bs = Lin_DE(params = params, bernstein = True, transpose=False)
f_pc = Lin_DE(params = params, bernstein = False, transpose=False)

#%%

x_0 = np.array([1])
h = 0.05
T = 15
alpha = 0.5


#%%
### Diethelm Predictor-Corrector ###

tic_pc = time.perf_counter_ns()
t, x_pc = ivp_diethelm(f_bs.f, x_0, alpha_vals = alpha, T = T, dt = h, return_t_vals=True)
toc_pc = time.perf_counter_ns()

time_pc = (toc_pc - tic_pc)*1e-9


#%%
### Bernstein Splines ###
N_it_vals = np.arange(1, 11)

x_bs = np.zeros([len(N_it_vals), len(t)])
time_bs = np.zeros(len(N_it_vals))

q = 1
CONV_TOL = 0

bs = BernsteinSplines(t, q) 
f_bs.set_bs_mult_upscale_functions(bs.splines_multiply, bs.splines_upscale)
spline_solver = bs.initialize_solver(f_bs.f, x_0, alpha, beta_vals=1)

i_it = 0
for N_it in N_it_vals:
    res_bs = spline_solver.run(method="local", verbose=False, conv_tol=CONV_TOL, conv_max_it=N_it)
    x_bs[i_it] = np.squeeze(res_bs['x'])
    time_bs[i_it] = res_bs['total_time']
    i_it+=1

#%%

x_analytical = x_0*ML(a*t**alpha, alpha, n_terms = 500)

print("\n~~~TABLE FOR ITERATION STEPS COMPARISON~~~\n")
error = np.abs(x_pc - x_analytical)
mean_error, sup_error = np.mean(error), np.max(error)
print(r" & mean error & sup error & total time (s) \\ \hline")
print_sc(fr"Diethelm PC & ${mean_error:.3e} }}$ & ${sup_error:.3e} }}$ & ${time_pc:.3e}}}$ \\ \hline")
for i in range(len(N_it_vals)):
    error = np.abs(x_bs[i] - x_analytical)
    mean_error, sup_error = np.mean(error), np.max(error)
    print_sc(fr"Splines $N = {N_it_vals[i]}$ &$ {mean_error:.3e} }}$& ${sup_error:.3e}}}$ & ${time_bs[i]:.3e}}}$ \\")
print(r'\hline'+"\n")

#%%
fig_size = 4.5
big_font, small_font = 12.5, 11.5
fig, axs = plt.subplots(1, 3, figsize=(3 * fig_size, fig_size), layout="tight")
#%%
base = 1/2
p_vals = np.arange(0,9)
h_vals = base**p_vals

colors, cmap = get_lin_line_colors(h_vals)

x_bs_h, x_pc_h = [None]*len(h_vals), [None]*len(h_vals)
error_bs_h, error_pc_h = [None]*len(h_vals), [None]*len(h_vals)
sup_error_bs_h, sup_error_pc_h = np.zeros(len(h_vals)), np.zeros(len(h_vals))
mean_error_bs_h, mean_error_pc_h = np.zeros(len(h_vals)), np.zeros(len(h_vals))
time_bs_h, time_pc_h = np.zeros(len(h_vals)), np.zeros(len(h_vals))
t_h = [None]*len(h_vals)

i_h = 0
CONV_TOL_DEFAULT = 1e-12
for h in h_vals:
    tic_pc = time.perf_counter_ns()
    t_h[i_h], x_pc_h[i_h] = ivp_diethelm(f_bs.f, x_0, alpha_vals = alpha, T = T, dt = h, return_t_vals=True)
    toc_pc = time.perf_counter_ns()

    time_pc_h[i_h] = (toc_pc - tic_pc)*1e-9

    label_str = rf"$h\,={1/base:.0f}^{{-{i_h}}}$"

    axs[0].plot(t_h[i_h], x_pc_h[i_h], label=fr'{label_str}', color=colors[i_h])

    bs = BernsteinSplines(t_h[i_h], q) 
    f_bs.set_bs_mult_upscale_functions(bs.splines_multiply, bs.splines_upscale)
    spline_solver = bs.initialize_solver(f_bs.f, x_0, alpha, beta_vals=1)
    res_bs = spline_solver.run(method="local", verbose=False, conv_tol=CONV_TOL_DEFAULT, conv_max_it=5000)
    x_bs_h[i_h] = np.squeeze(res_bs['x'])
    time_bs_h[i_h] = res_bs['total_time']

    axs[1].plot(t_h[i_h], x_bs_h[i_h], label=fr'{label_str}', color=colors[i_h])

    x_analytical_h = x_0*ML(a*t_h[i_h]**alpha, alpha, n_terms = 500)
    sup_error_pc_h[i_h] = np.max(np.abs(x_pc_h[i_h] - x_analytical_h))
    sup_error_bs_h[i_h] = np.max(np.abs(x_bs_h[i_h] - x_analytical_h))

    mean_error_pc_h[i_h] = np.mean(np.abs(x_pc_h[i_h] - x_analytical_h))
    mean_error_bs_h[i_h] = np.mean(np.abs(x_bs_h[i_h] - x_analytical_h))

    i_h += 1

axs[0].set_title('Diethelm PC')
axs[0].plot(t_h[-1], x_analytical_h, "--", label=r"Analytical", linewidth=2.5)
axs[0].legend()
axs[0].set_xlabel("$t$")
axs[0].set_ylabel('x')

axs[1].set_title('Splines')
axs[1].plot(t_h[-1], x_analytical_h, "--", label=r"Analytical", linewidth=2.5)
axs[1].legend()
axs[1].set_xlabel("$t$")
axs[1].set_ylabel('x')

axs[2].set_title('Error')
axs[2].loglog(h_vals, sup_error_pc_h, linewidth=2, label = 'Sup. error Diethelm PC', color='black')
axs[2].loglog(h_vals, sup_error_bs_h, linewidth=2, label = 'Sup. error Splines', color='royalblue')

axs[2].loglog(h_vals, mean_error_pc_h, linestyle='dashdot', linewidth=2, label = 'Mean error Diethelm PC', color='black')
axs[2].loglog(h_vals, mean_error_bs_h, linestyle='dashdot', linewidth=2, label = 'Mean error Splines', color='royalblue')
axs[2].invert_xaxis()
axs[2].legend()

axs[2].set_xlabel(r"$h$ (log, decreasing)")
axs[2].set_ylabel(r"Error (log)")

if SAVE_AS_PDF:
    plt.savefig(
        f'figures/diethelm_lin_comparison--alpha_{str(alpha).replace(".", "_")}.pdf',
        bbox_inches="tight",
    )
else:
    plt.show()
#%%

print("\n~~~TABLE FOR h COMPARISON ~~~\n")

print(r"$h$ & mean error PC & sup error PC & time PC (s) & mean error spl. & sup error spl. & time spl. (s)  \\ \hline")
for i_h in range(len(h_vals)):
    print_sc(fr"${1/base:.0f}^{{-{i_h}}}$ & ${mean_error_pc_h[i_h]:.3e} }} $ & ${sup_error_pc_h[i_h]:.3e} }} $ & ${time_pc_h[i_h]:.3e} }} $ & ${mean_error_bs_h[i_h]:.3e} }} $ & ${sup_error_bs_h[i_h]:.3e} }} $ & ${time_bs_h[i_h]:.3e} }} $" + r"\\")
print(r"\hline" +"\n")
  