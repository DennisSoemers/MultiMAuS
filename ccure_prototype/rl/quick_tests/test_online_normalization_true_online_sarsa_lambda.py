"""
@author Dennis Soemers
"""

import numpy as np


def true_normalization(features):
    return [features[0], features[1], 0.1 * features[2], 0.01 * features[3]]


def true_value(features):
    return 15 + features[1] * 2 + features[2] * -0.2 + features[3] * 0.01


if __name__ == '__main__':
    unnormalized_feature_vectors = [
        [1, 1, 20, 300],
        [1, 0.5, 30, -100],
        [1, -1.1, 50, 123],
        [1, 1.1, -18, -225],
        [1, 0.7, -40, 200]
    ]

    # this list is prior knowledge
    max_abs = np.asarray([1, 1.1, 50, 300])

    gamma = 0.99
    alpha = 0.001
    lamb = 0.8

    # first do a bit of training with normalizations based on prior knowledge
    print("Training with prior knowledge about maximum absolute feature values...")
    z_prior = np.asarray([0, 0, 0, 0])
    w_prior = np.asarray([0, 0, 0, 0])
    Q_old = 0
    raw_x = np.asarray(unnormalized_feature_vectors[0])
    x = raw_x / max_abs

    for t in [1, 2, 3, 4]:
        print("t = ", t)
        R = true_value(raw_x)
        raw_x_prime = np.asarray(unnormalized_feature_vectors[t])
        x_prime = raw_x_prime / max_abs
        Q = np.dot(w_prior, x)
        Q_prime = np.dot(w_prior, x_prime)
        delta = R + gamma*Q_prime - Q
        z_prior = gamma * lamb * z_prior + (1 - alpha * gamma * lamb * np.dot(z_prior, x)) * x
        w_prior = w_prior + alpha * (delta + Q - Q_old) * z_prior - alpha * (Q - Q_old) * x

        Q_old = Q_prime
        raw_x = raw_x_prime
        x = x_prime
        print("z = ", z_prior)
        print("w = ", w_prior)

    print("")

    # now exactly the same training, but with online normalization
    print("Training without prior knowledge about maximum absolute feature values...")
    z = np.asarray([0, 0, 0, 0])
    w = np.asarray([0, 0, 0, 0])
    w_old = np.copy(w)
    Q_old = 0
    raw_x = np.asarray(unnormalized_feature_vectors[0])
    bounds = np.abs(raw_x)
    x = raw_x / bounds

    for t in [1, 2, 3, 4]:
        print("t = ", t)
        R = true_value(raw_x)
        raw_x_prime = np.asarray(unnormalized_feature_vectors[t])

        old_bounds = np.copy(bounds)
        bounds = np.maximum(old_bounds, np.abs(raw_x_prime))
        bounds_correction = old_bounds / bounds
        z = np.multiply(z, bounds_correction)
        w = np.multiply(w, bounds_correction)
        x = np.multiply(x, bounds_correction)
        w_old = np.multiply(w_old, bounds_correction)
        Q_old = np.dot(w_old, x)

        x_prime = raw_x_prime / bounds
        Q = np.dot(w, x)
        Q_prime = np.dot(w, x_prime)
        delta = R + gamma * Q_prime - Q
        z = gamma * lamb * z + (1 - alpha * gamma * lamb * np.dot(z, x)) * x
        w_old = np.copy(w)
        w = w + alpha * (delta + Q - Q_old) * z - alpha * (Q - Q_old) * x

        Q_old = Q_prime
        raw_x = raw_x_prime
        x = x_prime
        print("z = ", z)
        print("w = ", w)
