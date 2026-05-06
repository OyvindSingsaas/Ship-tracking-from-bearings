import numpy as np

from utils.configs import sensor_A, sensor_B
from utils.particle import run_particle_filter


observations = np.column_stack((sensor_A, sensor_B))

filtered_states, covariances, weights_history = run_particle_filter(observations, particle_count=10000, resample_threshold=0.5)
np.save("results/particle_filtered_states_10000.npy", filtered_states)
np.save("results/particle_covariances_10000.npy", covariances)
np.save("results/particle_weights_history_10000.npy", weights_history)

filtered_states, covariances, weights_history = run_particle_filter(observations, particle_count=100, resample_threshold=0.5)
np.save("results/particle_filtered_states_100.npy", filtered_states)
np.save("results/particle_covariances_100.npy", covariances)
np.save("results/particle_weights_history_100.npy", weights_history)

# filtered_states, covariances, weights_history = run_particle_filter(observations, particle_count=10000, resample_threshold=0.5)
# np.save("results/particle_filtered_states_test.npy", filtered_states)
# np.save("results/particle_covariances_test.npy", covariances)
# np.save("results/particle_weights_history_test.npy", weights_history)