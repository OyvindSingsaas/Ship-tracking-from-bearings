import os
import numpy as np
import matplotlib.pyplot as plt
import utils.plot as plot
import utils.funcs as funcs
from utils.configs import delta, prior_mu, prior_sigma, A, Q, R, sensor_A_location, sensor_B_location, l_bound, u_bound, sensor_A, sensor_B



particle_weights_history_path_10000 = "results/particle_weights_history_10000.npy"
weights_history_particle_10000 = np.load(particle_weights_history_path_10000)


particle_weights_history_path_1000 = "results/particle_weights_history_100.npy"
weights_history_particle_1000 = np.load(particle_weights_history_path_1000)

particle_weights_history_path_test = "results/particle_weights_history_test.npy"
weights_history_particle_test = np.load(particle_weights_history_path_test)

effective_sample_sizes_1000 = np.array([1.0 / np.sum(w ** 2) for w in weights_history_particle_1000])/100
effective_sample_sizes_10000 = np.array([1.0 / np.sum(w ** 2) for w in weights_history_particle_10000])/10000
effective_sample_sizes_test = np.array([1.0 / np.sum(w ** 2) for w in weights_history_particle_test])/10000


plt.figure(figsize=(8, 6))
#plt.plot(effective_sample_sizes_test, marker='o', linestyle='-', color='green', label ='100000 Particles and resample_threshold=0.5')
plt.plot(effective_sample_sizes_1000[:10], marker='o', linestyle='-', color='orange', label ='100 Particles')
plt.plot(effective_sample_sizes_10000[:10], marker='o', linestyle='-', color='blue', label ='10000 Particles')
plt.title("Effective Sample Size Evolution")
plt.xlabel("Time Step")
plt.ylabel("Effective Sample Size")
plt.grid()
plt.legend()
plt.savefig("plots/effective_sample_size_evolution.png")
plt.close()