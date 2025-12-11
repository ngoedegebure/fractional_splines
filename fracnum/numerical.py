import numpy as np
from scipy.special import gamma
from mpmath import hyp1f2

def ivp_diethelm(f, u_0, alpha_vals, T, dt, return_t_vals = False):
    h = dt
    N = int(T/h)
    t_vals = np.linspace(0, T, N+1)

    u_0_array = np.array([u_0])
    f_u_0 = f(t_vals[0], u_0_array)

    a_vals, b_vals = {}, {}

    d = len(u_0)

    if np.array(alpha_vals).size == 1:
        alpha_vals = np.ones(d)*alpha_vals
    elif np.array(alpha_vals).size != d:
        print("ERROR! Either give one alpha or d (dimensions) alpha's!")
        assert 0

    for alpha in alpha_vals:
        b_vals[alpha] = np.zeros(N+1)
        a_vals[alpha] = np.zeros(N+1)
        for k in range(1, N+1):
            b_vals[alpha][k] = k**alpha - (k-1)**alpha
            a_vals[alpha][k] = (k+1)**(alpha+1) - 2 * k**(alpha+1) + (k-1)**(alpha+1)

    y = np.zeros([N+1, d])
    y[0, :] = u_0
    for j in range(1, N+1):
        f_prev_vals = f(t_vals[:j], y[:j, :])
        p = np.zeros(d)

        for dim in range(d):
            alpha = alpha_vals[dim]
            b = b_vals[alpha]
            a = a_vals[alpha]

            p[dim] = u_0[dim] + h**alpha / gamma(alpha+1) * np.flip(b[1:(j+1)]) @ f_prev_vals[:, dim]
        
        f_pred_vals = f(t_vals[j], np.array([p]))
        f_pred_vals = np.reshape(f_pred_vals, [f_pred_vals.shape[0], d])
        for dim in range(d):
            alpha = alpha_vals[dim]
            b = b_vals[alpha]
            a = a_vals[alpha]

            # breakpoint()

            y[j, dim] = u_0[dim] + h**alpha / gamma(alpha+2) * (
                f_pred_vals[:, dim] + ((j-1)**(alpha+1)-(j-1-alpha)*j**alpha)*f_u_0.T[dim]
                +
                np.flip(a[2:(j+1)]) @ f_prev_vals[1:, dim]
            )

    if return_t_vals:
        return t_vals, y
    else:
        return y

def sin_I_a(t, alpha, omega):
    # Computes the fractional integral from 0 to t of sin(omega * t)
    # NOTE: t has to be scalar since mpmath.hyp1f2 does not accept vectors
    # If it looks messy: it is. Obtained from Laplace transforming
    # However, quite battle tested as of now, works well and not even too slow!:)

    if alpha == 0:
        # To speed up the process for nonfractional case 
        # TODO: make general
        result = np.sin(omega*t)
    elif alpha == 1:
        # Idem
        result = -np.cos(omega*t)/omega + 1/omega
    else:
        # Define the first hypergeometric term
        term1 = hyp1f2(alpha / 2, 1 / 2, alpha / 2 + 1, -1 / 4 * omega**2 * t**2)

        # Define the second hypergeometric term
        term2 = hyp1f2(alpha / 2 + 1 / 2, 3 / 2, alpha / 2 + 3 / 2, -1 / 4 * omega**2 * t**2)

        # Define the full expression
        result = (t**alpha * ((np.sin(omega * t) * term1) / alpha -
                        (omega * t * np.cos(omega * t) * term2) / (alpha + 1))) / gamma(alpha)

    return result

def I_rl_rect_left(alpha_int, t_vals_int):
    N = len(t_vals_int)
    J = np.zeros([N, N])

    for n in range(1,N):
        h = t_vals_int[n] - t_vals_int[n-1]
        for j in range(n):
            J[n,j] = h**alpha_int / alpha_int * ((n - j)**alpha_int - (n-1-j)**alpha_int)
    J = J*1/gamma(alpha_int)

    return J

def build_hilf_knot_vals(eps, T, c, gamm, h_max, tol = 1e-10):
    if np.abs(gamm - 1) < tol:
        return np.linspace(0, T, int(T/h_max)+1)
    t_vals = np.array([eps])
    t = eps
    while t < T:
        h = (c**(1/(1-gamm))-1)*t
        if h < h_max:
            t = t + h
        else:
            t = t+ h_max
        t_vals = np.append(t_vals, t)
    return t_vals