import numpy as np
def get_matrix(alpha, beta, gamma, num_points):
    """Return the matrix for the internal energy minimization.
    # Arguments
        alpha: The alpha parameter.
        beta: The beta parameter.
        gamma: The gamma parameter.
        num_points: The number of points in the curve.
    # Returns
        The matrix for the internal energy minimization. (i.e. A + gamma * I)
    """
    a = 2 * alpha + 6 * beta
    b = -(alpha + 4 * beta)
    c = beta
    eye = np.eye(num_points)
    diag_mat_a = a * eye.copy()
    diag_mat_b = b * np.roll(eye.copy(), 1, 0) + b * np.roll(eye.copy(), -1, 0)
    diag_mat_c = c * np.roll(eye.copy(), 2, 0) + c * np.roll(eye.copy(), -2, 0)
    A = diag_mat_a + diag_mat_b + diag_mat_c
    return np.linalg.pinv(A +  gamma * np.eye(num_points))
