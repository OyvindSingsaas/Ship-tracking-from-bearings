import numpy as np
from utils.configs import sensor_A, sensor_B
from utils.ensemble import run_ensemble_kalman_filter

observations = np.column_stack((sensor_A, sensor_B))
filtered_states, covariances = run_ensemble_kalman_filter(observations, particle_count=1000)
np.save("results/ensemble_filtered_states_1000.npy", filtered_states)
np.save("results/ensemble_covariances_1000.npy", covariances)

filtered_states, covariances = run_ensemble_kalman_filter(observations, particle_count=100)
np.save("results/ensemble_filtered_states_100.npy", filtered_states)
np.save("results/ensemble_covariances_100.npy", covariances)