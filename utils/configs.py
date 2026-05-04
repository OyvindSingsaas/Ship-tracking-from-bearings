import numpy as np


sensor_A = np.loadtxt("data/sensorA.txt")
sensor_B = np.loadtxt("data/sensorB.txt")

sensor_A_location = np.array([0, 0])
sensor_B_location = np.array([40, 40])

l_bound = -5
u_bound = 45

#Sampling frequency delta
delta = 1/60
#Prior on state (Gaussian)
prior_mu = np.array([10, 30, 1, -1])  # [Et, Nt, vt, ut]
prior_sigma = np.diag([10**2, 10**2, 5**2, 5**2])  # Variances for each state component
#State transition matrix 
A = np.diag([1, 1, 1, 1])  
A[0, 2] = delta  # Et depends on vt
A[1, 3] = delta  # Nt depends on ut
#State transition noise covariance
Q = np.diag([0.1**2, 0.1**2, 0.5**2, 0.5**2])  # Process noise variances
#Measurement noise covariance
R = np.diag([0.1**2, 0.1**2])  # Measurement noise variances for angles from sensors A and B
