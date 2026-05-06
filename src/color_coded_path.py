import os
import numpy as np
import matplotlib.pyplot as plt
import utils.plot as plot
import utils.funcs as funcs
from utils.configs import delta, prior_mu, prior_sigma, A, Q, R, sensor_A_location, sensor_B_location, l_bound, u_bound, sensor_A, sensor_B

#Observed positions from angles
triangulated_paths = funcs.angle_to_position(sensor_A, sensor_B)
plot.plot_time_colored_path(triangulated_paths, path="results/observed_path_colored.png")
