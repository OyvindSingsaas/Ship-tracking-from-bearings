import numpy as np
from utils.configs import delta, prior_mu, prior_sigma, A, Q, R, sensor_A_location, sensor_B_location, l_bound, u_bound, sensor_A, sensor_B
import utils.funcs as funcs
from utils.funcs import wrap_to_pi

def observation_jacobian(state):
    position = state[:2]
    rel_A = position - sensor_A_location
    rel_B = sensor_B_location - position

    denom_A = rel_A[0] ** 2 + rel_A[1] ** 2
    denom_B = rel_B[0] ** 2 + rel_B[1] ** 2

    H = np.zeros((2, 4))
    H[0, 0] = rel_A[1] / denom_A
    H[0, 1] = -rel_A[0] / denom_A
    H[1, 0] = rel_B[1] / denom_B
    H[1, 1] = -rel_B[0] / denom_B
    return H

def one_step_extended_kalman_filter(observation, initial_state = prior_mu, initial_covariance = prior_sigma):
    state_dim = len(initial_state)
    state = initial_state
    covariance_estimate = initial_covariance

    # Prediction step
    state_predict = funcs.state_transition(state)
    covariance_predict = A @ covariance_estimate @ A.T + Q

    # Measurement update step
    H = observation_jacobian(state_predict)
    y_predict = funcs.measurement_function(state_predict)
    y_observed = observation
    innovation = wrap_to_pi(y_observed - y_predict)

    S = H @ covariance_predict @ H.T + R
    K = np.linalg.solve(S, H @ covariance_predict).T

    state_estimate = state_predict + K @ innovation
    identity = np.eye(state_dim)
    covariance_estimate = (identity - K @ H) @ covariance_predict @ (identity - K @ H).T + K @ R @ K.T
    covariance_estimate = 0.5 * (covariance_estimate + covariance_estimate.T)

    return state_estimate, covariance_estimate

def run_extended_kalman_filter(observations):
    T = len(observations)
    state_dim = len(prior_mu)
    states = np.zeros((T+1, state_dim))
    covariances = np.zeros((T+1, state_dim, state_dim))

    state = prior_mu
    covariance = prior_sigma
    states[0] = state
    covariances[0] = covariance
    for t in range(T):
        state, covariance = one_step_extended_kalman_filter(observations[t], state, covariance)
        states[t+1] = state
        covariances[t+1] = covariance
    return states, covariances

