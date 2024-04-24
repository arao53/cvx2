import numpy as np


def log_opt_return_sample(m):
    """Output an n x m matrix of samples from the log-normal mixture.

    Args:
        m (int): number of samples to generate

    Returns:
        ndarray: n x m matrix of samples from the log-normal mixture
    """
    n = 10  # dimension

    # means
    mu1 = np.array([0.4, 0.2, 0.4, 0.7, 0.9, 0.3, 0.9, 0.5, 0.5, 0.1])
    mu2 = np.array([5.1, 0.9, 0.7, 1.6, 2.2, 4.4, 3.2, 6.2, 4.2, 0.7])

    # covariance
    A = np.random.randn(n, n)
    B = np.random.randn(n, n)
    A = A @ A.T
    B = B @ B.T

    C1 = (
        np.diag(np.sqrt(np.diag(A) ** -1))
        @ A
        @ np.diag(np.sqrt(np.diag(A) ** -1))
    )
    C2 = (
        np.diag(np.sqrt(np.diag(B) ** -1))
        @ B
        @ np.diag(np.sqrt(np.diag(B) ** -1))
    )

    sigmas1 = np.array(
        [0.01, 0.03, 0.05, 0.04, 0.05, 0.09, 0.05, 0.01, 0.03, 0.12]
    )
    sigmas2 = np.array(
        [0.81, 0.31, 0.74, 0.91, 0.67, 0.71, 0.31, 0.42, 0.51, 0.41]
    )

    sigma1 = np.diag(sigmas1) @ C1 @ np.diag(sigmas1)
    sigma2 = np.diag(sigmas2) @ C2 @ np.diag(sigmas2)
    s1Half = np.linalg.cholesky(sigma1)
    s2Half = np.linalg.cholesky(sigma2)

    # Bernoulli sampling
    p = np.random.rand(m) < 0.9
    q = ~p

    # Samples from first distribution
    r1 = np.exp(np.tile(mu1, (m, 1)).T + s1Half @ np.random.randn(n, m))
    r1 = r1 * p[np.newaxis, :]

    # Samples from second distribution
    r2 = np.exp(np.tile(mu2, (m, 1)).T + s2Half @ np.random.randn(n, m))
    r2 = r2 * q[np.newaxis, :]

    # Return the sum of the two distributions
    return np.squeeze(r1 + r2)
