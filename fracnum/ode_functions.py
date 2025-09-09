from fracnum.splines.backend import np
from abc import ABC, abstractmethod


class ODEFunctions(ABC):
    def __init__(self, params, bernstein, transpose):
        self.params = params
        self.bernstein = bernstein
        self.transpose = transpose

        self.duplicate_dim = None

    def set_bs_mult_upscale_functions(self, bs_mult, bs_upscale):
        self.bs_mult, self.bs_upscale = bs_mult, bs_upscale

    @abstractmethod
    def f(self, t_vals, x_vals):
        pass


class VdP(ODEFunctions):
    def __init__(self, params={}, bernstein=False, transpose=False):
        super().__init__(params, bernstein, transpose)
        self.N_upscale = 3
        self.three_system = False

    def f(self, t_vals, x_vals):
        if self.transpose:
            x_vals = x_vals.T

        mu = self.params["mu"]
        if self.bernstein:
            # breakpoint()
            A_x_in, A_y_in = x_vals[0, :], x_vals[1, :]

            if self.three_system == False:
                # x' = y
                A_x_out = self.bs_upscale(A_y_in, self.N_upscale)

                # y' = mu * (1-x^2)y - x
                A_y_out = mu * self.bs_mult(
                    1 - self.bs_mult(A_x_in, A_x_in), A_y_in
                ) - self.bs_upscale(A_x_in, self.N_upscale)

                if "A" in self.params and "omega" in self.params:
                    if self.params["A"] != 0 and self.params["omega"] != 0:
                        # breakpoint()
                        A_y_out += self.params["A"] * np.sin(
                            self.params["omega"] * t_vals
                        )

                return np.array([A_x_out, A_y_out])
            else:
                A_z_in = x_vals[2, :]

                # x' = z
                A_x_out = self.bs_upscale(A_z_in, self.N_upscale)

                # y' = mu * (1-x^2)z - x
                A_y_out = mu * self.bs_mult(
                    1 - self.bs_mult(A_x_in, A_x_in), A_z_in
                ) - self.bs_upscale(A_x_in, self.N_upscale)

                # z' = y
                A_z_out = self.bs_upscale(A_y_in, self.N_upscale)

                return np.array([A_x_out, A_y_out, A_z_out])
        else:  # The non-Bernstein spline one is added for quicker comparison options with e.g. Diethelm's method (see fracnum.numerical)
            x, y = x_vals[0], x_vals[1]
            # x' = y
            x_out = y
            # y' = mu * (1-x^2)y - x
            y_out = -x - mu * (x**2 - 1) * y

            # Forcing included here to use in e.g. Diethelm or another numerical piece
            if "A" in self.params and "omega" in self.params:
                A = float(self.params["A"])
                omega = float(self.params["omega"])
                if A != 0 and omega != 0:
                    # TODO: CHANGE TO GENERAL ELEMENT ADDITION
                    y_out += A * np.sin(omega * t_vals)

            if "c" in self.params.keys():
                c = self.params["c"]
                if c != 0:
                    # TODO: CHANGE TO GENERAL ELEMENT ADDITION
                    y_out += c

            return np.array([x_out, y_out]).T


class NegExp(ODEFunctions):
    def __init__(self, params={}, bernstein=False, transpose=False):
        super().__init__(params, bernstein, transpose)
        self.N_upscale = 1

    def f(self, t_vals, x_vals):
        if self.transpose:
            x_vals = x_vals.T

        if self.bernstein:
            A_x_in = x_vals[0, :]

            A_x_out = -A_x_in

            return np.array([A_x_out])
        else:  # The non-Bernstein spline one is added for quicker comparison options with e.g. Diethelm's method (see fracnum.numerical)
            x, y = x_vals[0]

            x_out = x

            return np.array([x_out]).T


class t_k(ODEFunctions):
    def __init__(self, params={}, bernstein=False, transpose=False):
        super().__init__(params, bernstein, transpose)
        self.N_upscale = 1

    def f(self, t_vals, x_vals):
        k = np.array(self.params["k"])
        if "c" in self.params.keys():
            c = self.params["c"]
        else:
            c = 0
        res = t_vals**k + c

        if self.transpose:
            return res.T
        else:
            return res


class test_fun(ODEFunctions):
    def __init__(self, params={}, bernstein=False, transpose=False):
        super().__init__(params, bernstein, transpose)
        self.N_upscale = 1

    def f(self, t_vals, x_vals):
        return np.array([t_vals**2 - (1 - t_vals) ** 3])


class lin_damp_os(ODEFunctions):
    def __init__(self, params={}, bernstein=True, transpose=False):
        super().__init__(params, bernstein, transpose)
        self.N_upscale = 1

    def f(self, t_vals, x_vals):
        if self.transpose:
            x_vals = x_vals.T

        eta, omega = self.params["eta"], self.params["omega"]
        if self.bernstein:
            A_x_in, A_y_in = x_vals[0, :], x_vals[1, :]

            # x' = y
            A_x_out = A_y_in

            # y' = mu * (1-x^2)y - x
            A_y_out = -eta * A_y_in - omega**2 * A_x_in

            return np.array([A_x_out, A_y_out])
        else:
            assert 0


class Infiltration(ODEFunctions):
    def __init__(
        self,
        params={},
        bernstein=False,
        transpose=False,
        M=None,
        dz=None,
        b=None,
        T_z0=None,
        T_zb=None,
        P_z0=None,
        P_zb=None,
    ):

        super().__init__(params, bernstein, transpose)
        self.N_upscale = 2
        self.M = M  # number of discretization points
        self.dz = dz  # space discretization step
        self.b = b  # end of space interval
        self.T_z0 = lambda x: np.array([T_z0(x)])
        self.T_zb = lambda x: np.array([T_zb(x)])
        self.P_z0 = lambda x: np.array([P_z0(x)])
        self.P_zb = lambda x: np.array([P_zb(x)])

        self.k = float(self.params["k"])
        self.beta = float(self.params["beta"])
        self.chi = float(self.params["chi"])
        self.h = float(self.params["h"])
        self.alpha = float(self.params["alpha"])

    def f_T_int(self, T_vals, P_vals):
        k_part = self.k * self.bs_upscale(
            T_vals[0] - 2 * T_vals[1] + T_vals[2], self.N_upscale
        )
        beta_part = (
            self.beta / 4 * self.bs_mult(T_vals[2] - T_vals[0], P_vals[2] - P_vals[0])
        )
        chi_part = (
            self.chi / 4 * self.bs_mult(P_vals[2] - P_vals[0], P_vals[2] - P_vals[0])
        )
        # breakpoint()
        return (k_part + beta_part + chi_part) / (self.dz**2)

    def f_P_int(self, T_vals, P_vals):
        h_part = self.h * self.bs_upscale(
            P_vals[0] - 2 * P_vals[1] + P_vals[2], self.N_upscale
        )
        a_k_part = (
            self.alpha
            * self.k
            * self.bs_upscale(T_vals[0] - 2 * T_vals[1] + T_vals[2], self.N_upscale)
        )
        a_beta_part = (
            self.alpha
            * self.beta
            / 4
            * self.bs_mult(T_vals[2] - T_vals[0], P_vals[2] - P_vals[0])
        )
        a_chi_part = (
            self.alpha
            * self.chi
            / 4
            * self.bs_mult(P_vals[2] - P_vals[0], P_vals[2] - P_vals[0])
        )
        return (h_part + a_k_part + a_beta_part + a_chi_part) / (self.dz**2)

    def f(self, t_vals, U):
        N_dims = U.shape[0]
        N_half = N_dims // 2
        T_in, P_in = U[0:N_half], U[N_half::]
        T_out, P_out = [None] * N_half, [None] * N_half
        for i in range(N_half):
            if i == 0:
                # Start boundary point
                T_vals_sel = [self.T_z0(t_vals), T_in[i], T_in[i + 1]]
                P_vals_sel = [self.P_z0(t_vals), P_in[i], P_in[i + 1]]
            elif i == (N_half - 1):
                # End boundary point
                # breakpoint()
                T_vals_sel = [T_in[i - 1], T_in[i], self.T_zb(t_vals)]
                P_vals_sel = [P_in[i - 1], P_in[i], self.P_zb(t_vals)]
            else:
                # Internal point
                T_vals_sel = [T_in[i - 1], T_in[i], T_in[i + 1]]
                P_vals_sel = [P_in[i - 1], P_in[i], P_in[i + 1]]

            T_out[i] = self.f_T_int(T_vals_sel, P_vals_sel)
            P_out[i] = self.f_P_int(T_vals_sel, P_vals_sel)

        U_out = np.concatenate([T_out, P_out])
        return U_out
