import numpy as np
from utils.configs import delta, prior_mu, prior_sigma, A, Q, R, sensor_A_location, sensor_B_location, l_bound, u_bound, sensor_A, sensor_B


def angle_to_position(angles_A, angles_B):

    # dir_A/dir_B: shape (N, 2)
    dir_A = np.stack([np.sin(angles_A), np.cos(angles_A)], axis=1)
    dir_B = np.stack([-np.cos(angles_B), -np.sin(angles_B)], axis=1)

    # M: shape (N, 2, 2) — columns are dir_A and -dir_B
    M = np.stack([dir_A, -dir_B], axis=-1)
    rhs = sensor_B_location - sensor_A_location  # shape (2,)

    # np.linalg.solve broadcasts over N systems; t shape (N, 2)
    t = np.zeros((len(M), 2))  # Preallocate t
    for i in range(len(M)):
        t[i] = np.linalg.solve(M[i], rhs)

    positions = sensor_A_location + t[:, 0:1] * dir_A  # shape (N, 2)
    return positions


def position_to_angle(positions):
    dir_A = positions - sensor_A_location  # shape (N, 2)
    dir_B = positions - sensor_B_location  # shape (N, 2)

    angles_A = np.arctan2(dir_A[:, 0], dir_A[:, 1])  # shape (N,)
    angles_B = np.arctan2(-dir_B[:, 1], -dir_B[:, 0])  # shape (N,)

    return angles_A, angles_B


def state_transition(state, noise = False):
    if noise:
        process_noise = np.random.multivariate_normal(np.zeros(4), Q)
        return A @ state + process_noise
    return A @ state

def measurement_function(state, noise = False):
    position = state[:2]
    angles_A, angles_B = position_to_angle(position[None, :])
    if noise:
        measurement_noise = np.random.multivariate_normal(np.zeros(2), R)
        return np.concatenate([angles_A, angles_B]) + measurement_noise
    return np.concatenate([angles_A, angles_B])

def sample_initial_state():
    return np.random.multivariate_normal(prior_mu, prior_sigma)

def wrap_to_pi(angles):
    return (angles + np.pi) % (2 * np.pi) - np.pi
