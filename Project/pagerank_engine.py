import numpy as np

def compute_pagerank(adj_matrix, d=0.85, max_iter=100, tol=1e-6):
    """
    Computes PageRank using the Power Iteration method.

    Parameters:
        adj_matrix: Square 2D numpy array of edge weights.
        d: Damping factor (default 0.85).
        max_iter: Maximum power iterations.
        tol: L1 convergence threshold.

    Returns:
        r: Stationary PageRank probability distribution vector.
        iterations: Number of iterations to convergence.
        diff_history: L1 norm difference per iteration.
    """
    N = adj_matrix.shape[0]
    out_degrees = adj_matrix.sum(axis=1)

    # Stochastic transition matrix P
    P = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        if out_degrees[i] > 0:
            P[i, :] = adj_matrix[i, :] / out_degrees[i]
        else:
            P[i, :] = np.ones(N) / N  # Teleport from sink nodes

    # Initial state: Uniform distribution
    r = np.ones(N) / N
    diff_history = []

    for it in range(1, max_iter + 1):
        r_next = d * np.dot(r, P) + (1.0 - d) / N

        # Enforce sum to 1 to prevent floating point drift
        r_next /= np.sum(r_next)

        diff = np.linalg.norm(r_next - r, ord=1)
        diff_history.append(diff)

        r = r_next
        if diff < tol:
            return r, it, diff_history

    return r, max_iter, diff_history
