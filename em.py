"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n = X.shape[0]
    K = mixture.mu.shape[0]

    # Initialization
    log_posterior = np.float64(np.zeros((n, K)))
    sum_log_likelihood = 0

    # For each point i and cluster j:
    for i in range(n):
        # Boolean "is different from zero" for i'th row:
        is_not_zero = (X[i, :] != 0)
        for j in range(K):
            # Simplifying variables:
            mu = mixture.mu[j, is_not_zero]
            var = mixture.var[j]
            X_observed = X[i, is_not_zero]
            d = len(X_observed)
            # Log Gaussian:
            # log_likelihood = (-d/2) * np.log(2 * np.pi * var) - (1/2) * ((X_observed - mu) ** 2).sum() / var
            log_likelihood = (-d / 2) * np.log(2 * np.pi * var) - (1 / 2) * (np.sum((X_observed - mu) ** 2)) / var
            log_posterior[i, j] = np.log(mixture.p[j] + 1e-16) + log_likelihood
        # Sum over all K for for each i
        total = logsumexp(log_posterior[i, :])
        log_posterior[i, :] = log_posterior[i, :] - total
        sum_log_likelihood += total
    # Exponentiate the logarithm to take the posterior
    posterior = np.exp(log_posterior)
    return posterior, sum_log_likelihood



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    raise NotImplementedError


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
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


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    raise NotImplementedError