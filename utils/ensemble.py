import numpy as np
from utils.configs import delta, prior_mu, prior_sigma, A, Q, R, sensor_A_location, sensor_B_location, l_bound, u_bound, sensor_A, sensor_B
import utils.funcs as funcs
from utils.funcs import wrap_to_pi



def ensemble_mean(states):
    return np.mean(states, axis=0)

def ensemble_covariance(states, mean):
    centered_states = states - mean
    particle_count = states.shape[0]
    if particle_count < 2:
        return np.zeros((states.shape[1], states.shape[1]))
    return centered_states.T @ centered_states / (particle_count - 1)

def estimated_kalman_gain(predicted_particles, predicted_observations):
    particle_count = predicted_particles.shape[0]
    state_mean = ensemble_mean(predicted_particles)
    observation_mean = ensemble_mean(predicted_observations)

    centered_states = predicted_particles - state_mean
    centered_observations = wrap_to_pi(predicted_observations - observation_mean)

    cross_covariance = centered_states.T @ centered_observations / (particle_count - 1)
    observation_covariance = centered_observations.T @ centered_observations / (particle_count - 1)
    K = cross_covariance @ np.linalg.inv(observation_covariance + R)
    return K

def step_ensemble_kalman_filter(particles, observation):

    # Prediction
    predicted_particles = np.array([funcs.state_transition(particle, noise=True) for particle in particles])

    # Measurement update
    perturbed_observations = np.array([
        observation + np.random.multivariate_normal(np.zeros(2), R)
        for _ in range(len(predicted_particles))
    ])
    predicted_observations = np.array([funcs.measurement_function(p) for p in predicted_particles])
    innovations = wrap_to_pi(perturbed_observations - predicted_observations)
    K = estimated_kalman_gain(predicted_particles, predicted_observations)
    updated_particles = predicted_particles + innovations @ K.T
    ensemble_covariance_estimate = ensemble_covariance(updated_particles, ensemble_mean(updated_particles))
    return updated_particles, ensemble_covariance_estimate

def run_ensemble_kalman_filter(observations, particle_count=1000):
    particles = np.random.multivariate_normal(prior_mu, prior_sigma, size=particle_count)
    filtered_states = np.zeros((len(observations)+1, len(prior_mu)))
    filtered_covariances = np.zeros((len(observations)+1, len(prior_mu), len(prior_mu)))
    filtered_states[0] = ensemble_mean(particles)
    filtered_covariances[0] = ensemble_covariance(particles, filtered_states[0])

    for t in range(len(observations)):
        particles, covariance_estimate = step_ensemble_kalman_filter(particles, observations[t])
        state_estimate = ensemble_mean(particles)
         # Ensure covariance is symmetric
        covariance_estimate = 0.5 * (covariance_estimate + covariance_estimate.T)
        filtered_states[t+1] = state_estimate
        filtered_covariances[t+1] = covariance_estimate

    return filtered_states, filtered_covariances