import numpy as np
import matplotlib.pyplot as plt
from utils.configs import delta, prior_mu, prior_sigma, A, Q, R, sensor_A_location, sensor_B_location, l_bound, u_bound, sensor_A, sensor_B
import utils.funcs as funcs
from utils.extended import run_extended_kalman_filter

observations = funcs.angle_to_position(sensor_A, sensor_B)
observations = np.column_stack((sensor_A, sensor_B))
filtered_states, covariances = run_extended_kalman_filter(observations)
np.save("results/extended_filtered_states.npy", filtered_states)
np.save("results/extended_covariances.npy", covariances)