import numpy as np

from utils.configs import sensor_A, sensor_B
from utils.particle import run_particle_filter


observations = np.column_stack((sensor_A, sensor_B))
filtered_states, covariances, weights_history = run_particle_filter(observations, particle_count=10000, resample_threshold=0.0)
np.save("results/particle_filtered_states_10000.npy", filtered_states)
np.save("results/particle_covariances_10000.npy", covariances)
np.save("results/particle_weights_history_10000.npy", weights_history)

filtered_states, covariances, weights_history = run_particle_filter(observations, particle_count=1000, resample_threshold=0.0)
np.save("results/particle_filtered_states_1000.npy", filtered_states)
np.save("results/particle_covariances_1000.npy", covariances)
np.save("results/particle_weights_history_1000.npy", weights_history)