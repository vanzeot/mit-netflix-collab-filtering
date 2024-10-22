"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, d = X.shape
    K = mixture.mu.shape[0]

    # Initialization
    posterior = np.float64(np.zeros((n, K)))
    log_likelihood = 0

    # For each point i and cluster j:
    for i in range(n):
        for j in range(K):
            # Simplifying variables:
            mu = mixture.mu[j]
            var = mixture.var[j]
            # Gaussian:
            likelihood = 1 / ((2 * np.pi * var) ** (d / 2)) * np.exp(-np.divide(np.linalg.norm(X[i] - mu) ** 2, 2 * var))
            posterior[i, j] = mixture.p[j] * likelihood
        # Sum over all clusters (for each point i)
        total = posterior[i, :].sum()
        posterior[i, :] = posterior[i, :] / total
        # Take the logarithm
        log_likelihood += np.log(total)

    return posterior, log_likelihood


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
        for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    K = post.shape[1]

    # Clusters members
    estimated_members = np.sum(post, axis=0)
    mixture_weight = estimated_members / n

    # Initialization
    mu, var = np.zeros((K,d)), np.zeros(K)

    # For each cluster:
    for j in range(K):
        # (1/ñ) . Sum( delta(j|i) . x(i) )
        mu[j, :] = (1/estimated_members[j]) * (X * post[:, j, None]).sum(axis=0)
        # (1/ñ.d) . Sum( delta(j|i) . ||x(i)-mu(j)||² )
        var[j] = (((mu[j] - X)**2).sum(axis=1) @ post[:, j])/(d*estimated_members[j])
    return GaussianMixture(mu, var, mixture_weight)


def run(X: np.ndarray, mixture: GaussianMixture, post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
        for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
        for all components for all examples
        float: log-likelihood of the current assignment
    """
    log_likelihood = 0
    previous_log_likelihood = 0
    epsilon = 1e-6

    while log_likelihood - previous_log_likelihood >= epsilon * np.abs(log_likelihood) or previous_log_likelihood == 0:
        # Update previous log:
        previous_log_likelihood = log_likelihood
        # Take steps E and M:
        post, log_likelihood = estep(X, mixture)
        mixture = mstep(X, post)

    return mixture, post, log_likelihood
