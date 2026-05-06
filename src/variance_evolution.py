import os
import numpy as np
import matplotlib.pyplot as plt
import utils.plot as plot
import utils.funcs as funcs
from utils.configs import delta, prior_mu, prior_sigma, A, Q, R, sensor_A_location, sensor_B_location, l_bound, u_bound, sensor_A, sensor_B


# #Load filtered states and covariances from extended Kalman filter
covariances_extended = np.load("results/extended_covariances.npy")

particle_covariances_path_10000 = "results/particle_covariances_10000.npy"
covariances_particle_10000 = np.load(particle_covariances_path_10000)

particle_covariances_path_100 = "results/particle_covariances_100.npy"
covariances_particle_100 = np.load(particle_covariances_path_100)

ensemble_covariances_path_1000 = "results/ensemble_covariances_1000.npy"
covariances_ensemble_1000 = np.load(ensemble_covariances_path_1000)

ensemble_covariances_path_100 = "results/ensemble_covariances_100.npy"
covariances_ensemble_100 = np.load(ensemble_covariances_path_100)

plot.plot_covariance_traces(
	[
		covariances_extended[1:],
		covariances_particle_10000[1:],
		covariances_particle_100[1:],
		covariances_ensemble_1000[1:],
		covariances_ensemble_100[1:],
	],
	[
		"Extended Kalman Filter",
		"Particle Filter (10000)",
		"Particle Filter (100)",
		"Ensemble Kalman Filter (1000)",
		"Ensemble Kalman Filter (100)",
	],
	path="plots/variance_traces.png",
)