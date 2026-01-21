import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

total_time = 300  # time duration in seconds
fs = 4            # sampling frequency (Hz)

num_points = int(total_time * fs) + 1
t = np.linspace(0, total_time, num_points)

# Format: (time, value)
phi_data = np.array([
    [0, 0], [15, 40], [30, 40], [38, 55], [55, 55], [63, 45], 
    [78, 45], [85, 35], [102, 35], [110, 10], [120, 10], [140, 47], 
    [151, 47], [160, 33], [170, 33], [185, 62], [200, 62], [210, 42], 
    [225, 42], [235, 15], [250, 15], [265, 75], [277, 75], [290, 28], [300, 28]
])

gamma_data = np.array([
    [0, 0], [30, 320], [40, 320], [50, 180], [60, 180], [75, 240], 
    [85, 240], [102, 80], [110, 80], [150, 355], [155, 355], [160, 360], 
    [165, 360], [175, 350], [185, 350], [235, 0], [240, 0], [265, 260], 
    [272, 260], [282, 180], [287, 180], [297, 230], [300, 230]
])

t_phi_b_pts, val_phi_b_pts = phi_data[:, 0], phi_data[:, 1]
t_gamma_g_pts, val_gamma_g_pts = gamma_data[:, 0], gamma_data[:, 1]

def create_trajectory(t_pts, val_pts, t_query):
    interpolator = interp1d(
        t_pts,
        val_pts,
        kind="linear",
        bounds_error=False,
        fill_value=(val_pts[0], val_pts[-1]), 
        assume_sorted=True
    )
    return interpolator(t_query)

phi_b = create_trajectory(t_phi_b_pts, val_phi_b_pts, t)
gamma_g = create_trajectory(t_gamma_g_pts, val_gamma_g_pts, t)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(t, phi_b, linewidth=2, label=r"$\phi_b$", color='tab:blue')
ax1.set_ylabel("Angle [deg]") 
ax1.set_title("Control Trajectories")
ax1.grid(True, alpha=0.5)
ax1.legend(loc="upper right")

ax2.plot(t, gamma_g, linewidth=2, label=r"$\gamma_g$", color='tab:orange')
ax2.set_ylabel("Angle [deg]")
ax2.set_xlabel("Time [s]")
ax2.grid(True, alpha=0.5)
ax2.legend(loc="upper right")
ax2.set_ylim(-10, 380)
ax2.set_xlim(0, total_time)

plt.tight_layout()
plt.savefig('./Figure/trajectory_plot.png', dpi=150)
plt.close()

phi_b_deg_trajectory = phi_b[1:]
gamma_g_deg_trajectory = gamma_g[1:]

np.savez_compressed("./Data/trajectory.npz",
                    phi_b_deg_trajectory=phi_b_deg_trajectory,
                    gamma_g_deg_trajectory=gamma_g_deg_trajectory,
                    t=t[1:])
print("Saved to ./Data/trajectory.npz")