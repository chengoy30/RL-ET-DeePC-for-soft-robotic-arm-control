import numpy as np
from Lib.SoftArm_lib import constant_curvature

def generate_circular_trajectory(num_points=144, theta=np.pi/4, L_arm=93.0, total_steps=200, N=25):
    trajectory_base = np.zeros((num_points, 3))

    for i in range(num_points):
        phi = (2 * np.pi) / num_points * i
        x_r, y_r, z_r = constant_curvature(theta, phi, L_arm)
        trajectory_base[i, :] = [x_r, y_r, -z_r]

    window_size = N
    need_len = total_steps + window_size
    rep = int(np.ceil((need_len + 100) / num_points))
	
    trajectory_extended = np.tile(trajectory_base, (rep, 1))
    trajectory = trajectory_extended[:total_steps, :]
    return trajectory
