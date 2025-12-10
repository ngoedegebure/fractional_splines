import time
from scipy.special import gamma, betainc, beta
from tqdm import tqdm
from .backend import np


class SplineMethods:
    @staticmethod
    def build_total_t_vals(t_knot_vals, n, magnify=None):
        # Builds the coefficient t values for chosen knots and polynomial order
        # Returns both ordered per knot (first element) and a vector of all values (second element)

        t_knot_vals = np.asarray(t_knot_vals)

        if magnify is not None:
            knot_index, n_mag = magnify
            i_a, i_b = knot_index, knot_index + 1
            t_a, t_b = t_knot_vals[i_a], t_knot_vals[i_b]

            # breakpoint()
            new_magnified_vals = np.linspace(t_a, t_b, n_mag + 1)

            new_t_knot_vals = np.concatenate(
                [
                    t_knot_vals[(np.arange(len(t_knot_vals)) < i_a)],
                    new_magnified_vals[:-1],
                    t_knot_vals[(np.arange(len(t_knot_vals)) > i_a)],
                ]
            )

            t_knot_vals = new_t_knot_vals
            # print(f'magnified knot {knot_index}, ', new_magnified_vals)
        total_t_vals_ord = np.zeros(
            [len(t_knot_vals) - 1, n + 1]
        )  # Initialize the total t values

        for i in range(len(t_knot_vals) - 1):
            # Build ordered t values
            total_t_vals_ord[i, :] = np.linspace(
                t_knot_vals[i], t_knot_vals[i + 1], n + 1
            )

        # Flatten to vector and skip the "duplicate" final knot values
        t_eval_points = np.concatenate(
            [
                np.array([t_knot_vals[0]]),
                np.reshape(total_t_vals_ord[:, 1:], (len(t_knot_vals) - 1) * n),
            ]
        )

        return total_t_vals_ord, t_eval_points

    @staticmethod
    def I_a_b_beta(t_vals, alpha, k, bounds):
        # Computes the alpha-order integral of a polynomial s^k with support bounds = [a, b] on evaluation points t_vals

        a, b = bounds  # Get a and b for alias
        I_b = np.zeros(t_vals.shape)  # Initialize shape

        t_in_indices = t_vals > a  # Before a, all can be kept 0
        t_in = t_vals[t_in_indices]  # Values in the integration interval
        t_trans = (t_in - a) / (b - a)  # Compute transformation of t

        # Here the heavy lifting gets done: computes the influence of the fractional integral of one spline monomial
        I_b[t_in_indices] = (
            t_trans ** (alpha + k)
            * betainc(k + 1, alpha, np.fmin(1, t_trans) / t_trans)
            * beta(k + 1, alpha)
            / gamma(alpha)
        )

        return I_b

    @staticmethod
    def a_to_matrix(a_vector, n):
        if a_vector.size == 1:
            # If just one value, we are done
            total_a_matrix = np.reshape(a_vector, [1, 1])
        else:
            # Initialize matrix shape
            N_rows = int((len(a_vector) - 1) / max(n, 1))
            N_columns = n + 1
            total_a_matrix = np.zeros([N_rows, N_columns])

            # Build matrix
            total_a_matrix[:, :-1] = a_vector[:-1].reshape(
                [N_rows, N_columns - 1]
            )  # All columns except last
            total_a_matrix[:-1, -1] = total_a_matrix[
                1:, 0
            ]  # Last column except for last element
            total_a_matrix[-1, -1] = a_vector[-1]  # Last column last row

        return total_a_matrix

    @staticmethod
    def a_to_vector(total_a_matrix):
        # Initialize vector
        a_vector = np.zeros(total_a_matrix[:, :-1].size + 1)

        # Save t_k before reshaping
        last_val = total_a_matrix[-1, -1]

        # Build vector
        a_vector[:-1] = total_a_matrix[:, :-1].reshape(
            a_vector[:-1].shape
        )  # All values except for endpoints
        a_vector[-1] = last_val  # Final endpoint

        return a_vector

    @staticmethod
    def build_integral_basis(
        alpha,
        calc_vals_matrix,
        eval_vals_matrix,
        progress_verbose=True,
        time_verbose=True,
        dtype="float64",
    ):
        if alpha == 0:
            return None
        # Initialize basis function tensor of the shape:
        # B_I[knot_calc, order_calc, order_eval]
        B_I = np.zeros(
            [
                calc_vals_matrix.shape[0],
                calc_vals_matrix.shape[1],
                eval_vals_matrix.shape[0],
                eval_vals_matrix.shape[1],
            ],
            dtype=dtype,
        )

        # Two short hand aliases here
        n_knots = calc_vals_matrix.shape[0]
        n_eval = eval_vals_matrix.shape[1] - 1

        # Compute the eval functions as list
        t_eval_vals_list = SplineMethods.a_to_vector(eval_vals_matrix)

        # Get left and right points of knots and knot sizes (dt)
        t_a, t_b = calc_vals_matrix[:, 0], calc_vals_matrix[:, -1]
        dt = t_b - t_a

        # Transform evaluation points to 0 before interval, [0,1] on interval and lin. incr. after
        s_i = np.tile(t_eval_vals_list, (n_knots, 1))
        s_i = ((s_i.T - t_a) / dt).T

        time_basis, time_reshape = 0, 0
        disable_progress = not progress_verbose
        # breakpoint()
        for order_k in tqdm(
            range(calc_vals_matrix.shape[1]),
            desc=f"Building integral basis alpha = {alpha}",
            disable=disable_progress,
        ):
            ################################################
            ## Main work below: computes the basis values ##
            ################################################
            tic_c = time.time()
            B_I_vals_list = (
                dt**alpha * SplineMethods.I_a_b_beta(s_i, alpha, order_k, [0, 1]).T
            ).T
            toc_c = time.time()
            time_basis += toc_c - tic_c

            # Reshape into matrix values. Takes a bit of time but still more efficient than doing the loop before
            # TODO (possibly) remove for loop in this reshaping bit!
            tic_r = time.time()
            for calc_knot in range(n_knots):
                B_I[calc_knot, order_k, :, :] = SplineMethods.a_to_matrix(
                    B_I_vals_list[calc_knot, :], n_eval
                )
            toc_r = time.time()
            time_reshape += toc_r - tic_r

        if time_verbose:
            print(
                f"~ Integral basis alpha = {alpha} finished. Calculations: {time_basis:.4f} s. Reshaping: {time_reshape:.4f} s"
            )
        return B_I
