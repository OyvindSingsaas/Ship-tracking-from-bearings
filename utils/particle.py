import numpy as np
from utils.configs import delta, prior_mu, prior_sigma, A, Q, R, sensor_A_location, sensor_B_location, l_bound, u_bound, sensor_A, sensor_B
import utils.funcs as funcs
import matplotlib.pyplot as plt
from utils.funcs import wrap_to_pi



def sample_initial_particles(B):
    return np.random.multivariate_normal(prior_mu, prior_sigma, size=B)


def predict_particles(particles):
    process_noise = np.random.multivariate_normal(np.zeros(4), Q, size=particles.shape[0])
    return (A @ particles.T).T + process_noise


def compute_weights(particles_predicted, observation):
    predicted_observations = np.array([funcs.measurement_function(p) for p in particles_predicted])
    innovations = wrap_to_pi(observation - predicted_observations)
    precision = np.linalg.inv(R)
    log_weights = -0.5 * np.einsum('bi,ij,bj->b', innovations, precision, innovations)
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    weight_sum = np.sum(weights)
    if weight_sum == 0 or not np.isfinite(weight_sum):
        return np.full(particles_predicted.shape[0], 1.0 / particles_predicted.shape[0])
    return weights / weight_sum


def effective_sample_size(weights):
    return 1.0 / np.sum(weights ** 2)


def systematic_resample(particles, weights):
    particle_count = particles.shape[0]
    positions = (np.random.rand() + np.arange(particle_count)) / particle_count
    cumulative_weights = np.cumsum(weights)
    indices = np.searchsorted(cumulative_weights, positions)
    resampled_particles = particles[indices]
    resampled_weights = np.full(particle_count, 1.0 / particle_count)
    return resampled_particles, resampled_weights


def estimate_state(particles, weights):
    state_estimate = np.average(particles, axis=0, weights=weights)
    centered_particles = particles - state_estimate
    covariance_estimate = centered_particles.T @ (centered_particles * weights[:, None])
    covariance_estimate = 0.5 * (covariance_estimate + covariance_estimate.T)
    return state_estimate, covariance_estimate


def one_step_particle_filter(observation, particles, weights=None, resample_threshold=0.5):
    particle_count = particles.shape[0]
    if weights is None:
        weights = np.full(particle_count, 1.0 / particle_count)

    predicted_particles = predict_particles(particles)
    likelihood_weights = compute_weights(predicted_particles, observation)
    updated_weights = weights * likelihood_weights
    updated_weights_sum = np.sum(updated_weights)
    if updated_weights_sum == 0 or not np.isfinite(updated_weights_sum):
        updated_weights = np.full(particle_count, 1.0 / particle_count)
    else:
        updated_weights = updated_weights / updated_weights_sum

    state_estimate, covariance_estimate = estimate_state(predicted_particles, updated_weights)

    if effective_sample_size(updated_weights) < resample_threshold * particle_count:
        print("Resampling particles...")
        predicted_particles, updated_weights = systematic_resample(predicted_particles, updated_weights)

    return predicted_particles, updated_weights, state_estimate, covariance_estimate


def run_particle_filter(observations, particle_count=10000, resample_threshold=0.5):
    state_dim = len(prior_mu)
    observation_count = len(observations)

    filtered_states = np.zeros((observation_count + 1, state_dim))
    filtered_covariances = np.zeros((observation_count + 1, state_dim, state_dim))
    weights_history = np.zeros((observation_count + 1, particle_count))

    particles = sample_initial_particles(particle_count)
    weights = np.full(particle_count, 1.0 / particle_count)
    weights_history[0] = weights
    initial_state, initial_covariance = estimate_state(particles, weights)
    filtered_states[0] = initial_state
    filtered_covariances[0] = initial_covariance

    for t, observation in enumerate(observations, start=1):
        particles, weights, state_estimate, covariance_estimate = one_step_particle_filter(
            observation,
            particles,
            weights,
            resample_threshold=resample_threshold,
        )
        filtered_states[t] = state_estimate
        filtered_covariances[t] = covariance_estimate
        weights_history[t] = weights

    return filtered_states, filtered_covariances, weights_history

