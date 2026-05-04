import os
import numpy as np
import matplotlib.pyplot as plt
import utils.plot as plot
import utils.funcs as funcs
from utils.configs import delta, prior_mu, prior_sigma, A, Q, R, sensor_A_location, sensor_B_location, l_bound, u_bound, sensor_A, sensor_B

#Observed positions from angles
triangulated_paths = funcs.angle_to_position(sensor_A, sensor_B)
plot.plot_path(triangulated_paths, path="results/observed_path.png")

#Load filtered states and covariances from extended Kalman filter
filtered_states_exteended = np.load("results/extended_filtered_states.npy")
covariances_extended = np.load("results/extended_covariances.npy")
#Extract positions and variances
estimated_positions_extended = filtered_states_exteended[:, :2]
estimated_velocities_extended = filtered_states_exteended[:, 2:]
#Plot extended Kalman filter results with variance
plot.plot_path(estimated_positions_extended, variance=covariances_extended, path="results/extended_path.png")

particle_states_path_10000 = "results/particle_filtered_states_10000.npy"
particle_covariances_path_10000 = "results/particle_covariances_10000.npy"
particle_weights_history_path_10000 = "results/particle_weights_history_10000.npy"

filtered_states_particle_10000 = np.load(particle_states_path_10000)
covariances_particle_10000 = np.load(particle_covariances_path_10000)
weights_history_particle_10000 = np.load(particle_weights_history_path_10000)
estimated_positions_particle_10000 = filtered_states_particle_10000[:, :2]
plot.plot_path(estimated_positions_particle_10000, variance=covariances_particle_10000, path="results/particle_path_10000.png")
plot.plot_particle_weights_histogram(weights_history_particle_10000[:, 0], title="Particle Weight (First Particle) (10000 Particles)", path="results/particle_weights_histogram_first_10000.png")
plot.plot_particle_weights_histogram(weights_history_particle_10000[:, -1], title="Particle Weight (Last Particle) (10000 Particles)", path="results/particle_weights_histogram_last_10000.png")

particle_states_path_1000 = "results/particle_filtered_states_1000.npy"
particle_covariances_path_1000 = "results/particle_covariances_1000.npy"
particle_weights_history_path_1000 = "results/particle_weights_history_1000.npy"

filtered_states_particle_1000 = np.load(particle_states_path_1000)
covariances_particle_1000 = np.load(particle_covariances_path_1000)
weights_history_particle_1000 = np.load(particle_weights_history_path_1000)
estimated_positions_particle_1000 = filtered_states_particle_1000[:, :2]
plot.plot_path(estimated_positions_particle_1000, variance=covariances_particle_1000, path="results/particle_path_1000.png")
plot.plot_particle_weights_histogram(weights_history_particle_1000[:, 0], title="Particle Weight (First Particle) (1000 Particles)", path="results/particle_weights_histogram_first_1000.png")
plot.plot_particle_weights_histogram(weights_history_particle_1000[:, -1], title="Particle Weight (Last Particle) (1000 Particles)", path="results/particle_weights_histogram_last_1000.png")


ensemble_states_path_1000 = "results/ensemble_filtered_states_1000.npy"
ensemble_covariances_path_1000 = "results/ensemble_covariances_1000.npy"
filtered_states_ensemble = np.load(ensemble_states_path_1000)
covariances_ensemble_1000 = np.load(ensemble_covariances_path_1000)
estimated_positions_ensemble = filtered_states_ensemble[:, :2]
plot.plot_path(estimated_positions_ensemble, variance=covariances_ensemble_1000, path="results/ensemble_path_1000.png")


ensemble_states_path_100 = "results/ensemble_filtered_states_100.npy"
ensemble_covariances_path_100 = "results/ensemble_covariances_100.npy"
filtered_states_ensemble = np.load(ensemble_states_path_100)
covariances_ensemble_100 = np.load(ensemble_covariances_path_100)
estimated_positions_ensemble = filtered_states_ensemble[:, :2]
plot.plot_path(estimated_positions_ensemble, variance=covariances_ensemble_100, path="results/ensemble_path_100.png")
