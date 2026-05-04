import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Ellipse
from utils.configs import delta, prior_mu, prior_sigma, A, Q, R, sensor_A_location, sensor_B_location, l_bound, u_bound, sensor_A, sensor_B


def covariance_to_ellipse(position_covariance, n_std=1.0):
    eigenvalues, eigenvectors = np.linalg.eigh(position_covariance)
    eigenvalues = np.clip(eigenvalues, a_min=0.0, a_max=None)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * n_std * np.sqrt(eigenvalues)
    return width, height, angle


def plot_animated_path(observations, variance=None, path="results/extended_path.gif", interval=120):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(l_bound, u_bound)
    ax.set_ylim(l_bound, u_bound)
    ax.set_xlabel("East-West (km)")
    ax.set_ylabel("North-South (km)")

    ax.scatter(sensor_A_location[0], sensor_A_location[1], c='red', label='Sensor A')
    ax.scatter(sensor_B_location[0], sensor_B_location[1], c='blue', label='Sensor B')

    path_line, = ax.plot([], [], color='green', linewidth=1.5, alpha=0.8, label='Estimated Path')
    sensor_a_ray, = ax.plot([], [], linestyle=':', color='red', linewidth=1.2, alpha=0.35)
    sensor_b_ray, = ax.plot([], [], linestyle=':', color='blue', linewidth=1.2, alpha=0.35)
    current_point = ax.scatter([], [], c='green', s=35)
    ellipse = None

    def update(frame):
        nonlocal ellipse

        current_positions = observations[: frame + 1]
        path_line.set_data(current_positions[:, 0], current_positions[:, 1])
        current_point.set_offsets(observations[frame])
        sensor_a_ray.set_data(
            [sensor_A_location[0], observations[frame, 0]],
            [sensor_A_location[1], observations[frame, 1]],
        )
        sensor_b_ray.set_data(
            [sensor_B_location[0], observations[frame, 0]],
            [sensor_B_location[1], observations[frame, 1]],
        )

        if ellipse is not None:
            ellipse.remove()
            ellipse = None

        if variance is not None:
            position_covariance = variance[frame][:2, :2]
            width, height, angle = covariance_to_ellipse(position_covariance)
            ellipse = Ellipse(
                xy=observations[frame],
                width=width,
                height=height,
                angle=angle,
                edgecolor='green',
                facecolor='none',
                linewidth=1.5,
                alpha=0.35,
            )
            ax.add_patch(ellipse)

        artists = (path_line, sensor_a_ray, sensor_b_ray, current_point)
        return artists if ellipse is None else artists + (ellipse,)

    ax.legend()
    animation = FuncAnimation(fig, update, frames=len(observations), interval=interval, blit=False, repeat=False)
    animation.save(path, writer=PillowWriter(fps=max(1, int(round(1000 / interval)))))
    plt.close(fig)


def plot_path(observations, variance = None, path="results/extended_path.png"):

    plt.figure(figsize=(8, 8))
    plt.xlim(l_bound, u_bound)
    plt.ylim(l_bound, u_bound)
    plt.scatter(sensor_A_location[0], sensor_A_location[1], c='red', label='Sensor A')
    plt.scatter(sensor_B_location[0], sensor_B_location[1], c='blue', label='Sensor B')
    plt.scatter(observations[:, 0], observations[:, 1], c='green', alpha = 0.8, label='Estimated Path')
    if variance is not None:
        for position, covariance in zip(observations, variance):
            position_covariance = covariance[:2, :2]
            width, height, angle = covariance_to_ellipse(position_covariance)
            ellipse = Ellipse(
                xy=position,
                width=width,
                height=height,
                angle=angle,
                edgecolor='green',
                facecolor='none',
                linewidth=1,
                alpha=0.35,
            )
            plt.gca().add_patch(ellipse)
    plt.xlabel("East-West (km)")
    plt.ylabel("North-South (km)")
    plt.legend()
    plt.savefig(path)
    plt.close()
    
def plot_particle_weights_histogram(weights, title="Particle Weights", path="results/particle_weights_histogram.png"):
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(weights)), weights, alpha=0.7, color='blue')
    plt.title(title)
    plt.xlabel("Weight")
    plt.ylabel("Density")
    plt.grid()
    plt.savefig(path)
    plt.close()