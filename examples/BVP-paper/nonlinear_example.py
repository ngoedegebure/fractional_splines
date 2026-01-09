import numpy as np
import fracnum as fr
import matplotlib.pyplot as plt
from fracnum.plotting_utils import get_lin_line_colors
from scipy.special import gamma, betainc
from scipy.special import beta as betafun
from fracnum.numerical import build_hilf_knot_vals
from fracnum.ode_functions import ODEFunctions
from scipy.optimize import root

class Function(ODEFunctions):
    def __init__(self, params = {}, bernstein=False, transpose = False):
        super().__init__(params, bernstein, transpose)
        self.N_upscale = 1
    
    def f(self, t, x_vals):
        x = x_vals[0, :]

        x_out = np.cos(x*t*np.pi*4)/np.pi*2

        return np.array([x_out])
    
def xi(t, T, alpha, beta):
    gamm = alpha + beta - alpha*beta
    zeta = 1-gamm+alpha

    I_1 = gamma(gamm)*t**alpha/gamma(gamm+alpha)
    J_1 = (1-gamm-alpha)*betainc(gamm, zeta, t/T)*betafun(gamm, zeta)
    J_2 = zeta*betainc(zeta, gamm, 1-t/T)*betafun(zeta, gamm)

    return I_1 + (t/T)**zeta*T**alpha/gamma(alpha+1)*(J_1+J_2)

def omega(t, t_,T, alpha, beta, q=1):
    gamm = alpha + beta - alpha*beta
    zeta = 1-gamm+alpha

    if beta != 1:
        t_t_part = (t_/t)**(1-gamm)
    else:
        t_t_part = 1
    O_1 = (t_ - t)**alpha/gamma(alpha+1) * (alpha*betafun(gamm, alpha)+2*t_t_part)
    O_2 = (t_ - t)**zeta * T**alpha/(gamma(alpha+1)*T**zeta)*zeta*betafun(gamm, zeta)
    return O_1 + O_2

T = 0.5

q = 1

x_0 = np.array([1])
alpha = 0.75
eps = 10**(-10)
c = 3/2
h_max = 0.01

beta_vals = np.linspace(0, 1, 401)
Xi_vals = np.zeros(beta_vals.shape)
Omega_vals = np.zeros(beta_vals.shape)
i=0
for beta_sel in beta_vals:
    gamm = alpha + beta_sel - alpha*beta_sel
    t_knot_vals = build_hilf_knot_vals(eps, T, c, gamm, h_max)[:-1]
    Xi_vals[i]=np.max(xi(t_knot_vals, T, alpha, beta_sel))
    omega_t_vals = [omega(t_knot_vals[i], t_knot_vals[i+1],T, alpha, beta_sel) for i in range(len(t_knot_vals)-1)]
    Omega_vals[i] = np.max(omega_t_vals)
    i+=1

plt.plot(beta_vals, Xi_vals+Omega_vals, label =r'$\Xi + \Omega_\mathcal{A}^q$', linewidth=2.5, linestyle='--', color='black')
plt.plot(beta_vals, Xi_vals, label ='$\Xi$')
plt.plot(beta_vals, Omega_vals, label =r'$\Omega_\mathcal{A}^q$')
plt.plot(beta_vals, np.ones(len(Omega_vals)), color = 'red', linestyle="--", linewidth = 2.5, label = 'Convergence requirement upper bound')
plt.legend()
plt.xlabel(rf"$\beta$")
plt.ylabel(rf"$\rho$")
plt.savefig(f"figures/convergence_bound.pdf", bbox_inches='tight')
plt.close()
print('Upper bound:', np.max(Xi_vals+Omega_vals))

fun = Function(bernstein = True, transpose = False)

TOL = 1e-12
max_it = 800

N = 5+1
beta_vals = np.linspace(0, 1, N)
beta_colors, cmap = get_lin_line_colors(beta_vals)
t_hr_eval = np.linspace(eps/T,1, 3000)*T
i = 0
res_vals = [None]*N
for beta in beta_vals:
    gamm = alpha+beta-alpha*beta

    t_knot_vals = build_hilf_knot_vals(eps, T, c, gamm, h_max)
    bs = fr.splines.BernsteinSplines(t_knot_vals, fun.N_upscale*q, silent_mode = True, n_eval = q)

    fun.bs_mult, fun.bs_upscale = bs.splines_multiply, bs.splines_upscale

    solver = bs.initialize_solver(fun.f, x_0, alpha, beta_vals=beta)

    res = solver.run(verbose=False, conv_tol = TOL, conv_max_it=max_it, method='global', bvp=True, T = T, t_eval = t_hr_eval, save_x = False)
    res_vals[i]  =res
    print(f"beta: {beta}, delta: {res['delta']}")
    x, t, run_time_s = np.squeeze(res['x']), res['t'], res['total_time']
    label_str = fr"$\beta = {beta:.1f}$"
    plt.plot(t_hr_eval, x, color = beta_colors[i], label = fr"$\beta = {beta:.2f}$")
    plt.ylim([0,5])
    i+=1
        
plt.legend()
plt.xlabel("$t$"), plt.ylabel("$x$")
plt.savefig('figures/nonlin_ex_multiple_beta.pdf', bbox_inches='tight')
plt.close()

print(r'$\beta$ & knots & $\Delta_T(\tilde{x}_0)$ & it.&  avg. it. per knot \\ \hline')
for i in range(N):
    gamm = alpha+beta_vals[i] - alpha*beta_vals[i]
    t = build_hilf_knot_vals(eps, T, c, gamm, h_max)[:-1]
    N = len(t)
    output_str = rf"${beta_vals[i]:.2f}$ & ${N}$& ${np.squeeze(res_vals[i]['delta']):.3f}$&  ${res_vals[i]['n_it']}$& ${res_vals[i]['n_it_per_knot']:.3f}$"+ rf"\\ "
    
    print(output_str.replace('e+0', ' \cdot 10^').replace('e-0', ' \cdot 10^{-'))

beta= 0.5
gamm = alpha+beta - alpha*beta
t_knot_vals = build_hilf_knot_vals(eps, T, c, gamm, h_max)

gridlims, grid_dt = [0.005, T], 0.005

METHOD = 'GRID' # 'GRID', 'RF'

T_bvp_0 = T

TOL = 1e-12
max_it = 800

fun = Function(bernstein = True, transpose = False)

bs = fr.splines.BernsteinSplines(t_knot_vals, fun.N_upscale*q, silent_mode = True, n_eval = q)

fun.bs_mult, fun.bs_upscale = bs.splines_multiply, bs.splines_upscale

solver = bs.initialize_solver(fun.f, x_0, alpha, beta_vals=beta)
t_hr_eval = (np.linspace(eps/T,1, 3000))**1*T

if METHOD == 'RF':
    def root_function(T_bvp):
        res_rf = solver.run(verbose=False, conv_tol =TOL, conv_max_it=max_it, method='global', bvp=True, T = T_bvp[0], save_x=False)
        return res_rf['delta']

    sol = root(root_function, T_bvp_0, method = 'lm')
    print(sol)
    T_result = sol.x
elif METHOD == 'GRID':
    T_i_vals = np.linspace(gridlims[0], gridlims[1], int((gridlims[1]-gridlims[0])/grid_dt)+1)
    delta_vals = np.zeros(T_i_vals.shape)
    delta_min, i_min = np.inf, 0
    for i in range(len(T_i_vals)):
        res_gr = solver.run(verbose=False, conv_tol =TOL, conv_max_it=max_it, method='global', bvp=True, T=T_i_vals[i], save_x=False, t_eval = t_hr_eval)
        delta_vals[i] = np.squeeze(res_gr['delta'])
        if np.abs(delta_vals[i]) < delta_min:
            i_min = i
            delta_min = np.abs(delta_vals[i])
    T_result = [T_i_vals[i_min]]
else:
    T_result = np.array([T_bvp_0])

res = solver.run(verbose=False, conv_tol = TOL, conv_max_it=max_it, method='global', bvp=False, T = T_result[0], t_eval = t_hr_eval)

print(f"T: {T_result}, delta: {res['delta']}")
x, t, run_time_s = np.squeeze(res['x']), res['t'], res['total_time']
label_str = fr"$\beta = {beta:.1f}$"
plt.ylim([-1,5])
if METHOD == 'GRID':
    plt.plot(T_i_vals, delta_vals, color = 'black', label=r"$\Delta_T(\tilde{x}_{\,0})$")
plt.plot(t_hr_eval, x, label = r"$x^*$")
    
plt.vlines(T_result, ymin= -1, ymax =5, color = 'red', linestyle="--", linewidth = 1, label=rf"$T^* = {T_result[0]:.3f}$")
plt.legend()
plt.xlabel("$t$")
plt.savefig(f"figures/polynomial_delta_0_beta_{str(beta).replace('.', '_')}.pdf", bbox_inches='tight')
plt.close()