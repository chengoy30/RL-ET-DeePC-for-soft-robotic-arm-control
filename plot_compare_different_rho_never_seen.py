import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# IEEE single-column width: 3.5 in
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'lines.linewidth': 1.0,
})

def position_to_angle(x, y, z):
    # Handle special case when arm is straight (x=0, y=0)
    x_plane = np.sqrt(x**2 + y**2)

    if np.abs(x_plane) < 1e-6:
        # Arm is nearly straight
        theta = 0.0
        phi = 0.0
    else:
        # Calculate bending direction (phi/gamma_r)
        phi = np.arctan2(y, x)
        theta = 2.0 * np.arctan2(x_plane, -z)

        # Wrap theta to [0, 2π) range (0-360 degrees)
        theta = theta % (2 * np.pi)

    return theta, phi

rhos = [0.1, 0.5, 1.0]
colors = ['#e74c3c', '#3498db', '#2ca02c']
data_all = {}

for rho in rhos:
    d = np.load(f'./Saved_Testing_Data/ppo_never_seen_test_data_rho_{rho}.npz')
    data_all[rho] = d

ref_trajectory = data_all[rhos[0]]['ref_trajectory']

# Convert XYZ trajectories to theta-phi (bending angle and direction)
ref_angles = np.array([position_to_angle(x, y, z) for x, y, z in ref_trajectory])
ref_theta = ref_angles[:, 0]  # phi_b (bending angle)
ref_phi = ref_angles[:, 1]    # gamma_r (bending direction)

# Convert actual trajectories to angles for each rho
angles_all = {}
for rho in rhos:
    y_actual = data_all[rho]['y_actual']
    angles = np.array([position_to_angle(x, y, z) for x, y, z in y_actual])
    angles_all[rho] = angles

fig, axes = plt.subplots(5, 1, figsize=(3.5, 4.9), sharex=True,
                         gridspec_kw={'height_ratios': [2, 2, 1, 1, 1]})

# --- Top 2 subplots: tracking trajectory for phi_b (bending angle) and gamma_r (bending direction) ---
angle_labels = [r'$\phi_b$ (Bending Angle)', r'$\gamma_r$ (Bending Direction)']
angle_units = ['rad', 'rad']
ref_angle_data = [ref_theta, ref_phi]

for i in range(2):
    ax = axes[i]
    ax.plot(ref_angle_data[i], color='black', label='Reference')
    for rho, color in zip(rhos, colors):
        angle_actual = angles_all[rho][:, i]
        ax.plot(angle_actual, color=color, label=f'ρ={rho}')
    ax.set_ylabel(f'{angle_labels[i]} ({angle_units[i]})')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

# --- Bottom 3 subplots: trigger (action) for each rho ---
for j, (rho, color) in enumerate(zip(rhos, colors)):
    ax = axes[2 + j]  # Changed from axes[3 + j] to axes[2 + j] since we now have 2 angle plots instead of 3
    action_data = data_all[rho]['action_data']
    time_steps = np.arange(len(action_data))
    ax.bar(time_steps, action_data, color=color, alpha=0.8, width=0.5)
    ax.set_ylabel('Trigger')
    ax.set_yticks([0, 1])
    ax.set_ylim(-0.1, 1.3)
    ax.text(0.98, 0.85, f'ρ={rho}', transform=ax.transAxes,
            ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.2))
    ax.grid(True, alpha=0.3, axis='y')

axes[-1].set_xlabel('Step')

plt.tight_layout()
plt.savefig('./Figure/ppo_angle_tracking_and_trigger_never_seen.png', dpi=300, bbox_inches='tight')
plt.savefig('./Figure/ppo_angle_tracking_and_trigger_never_seen.pdf', bbox_inches='tight')
plt.show()
