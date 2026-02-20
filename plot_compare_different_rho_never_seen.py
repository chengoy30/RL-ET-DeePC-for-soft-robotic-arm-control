import numpy as np
import matplotlib.pyplot as plt
import matplotlib

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
    x_plane = np.sqrt(x**2 + y**2)

    if np.abs(x_plane) < 1e-6:
        theta = 0.0
        phi = 0.0
    else:
        phi = np.arctan2(y, x)
        theta = 2.0 * np.arctan2(x_plane, -z)

        theta = theta % (2 * np.pi)

    return theta, phi

rhos = [0.1, 0.5, 1.0]
colors = ['#e74c3c', '#3498db', '#2ca02c']
data_all = {}

for rho in rhos:
    d = np.load(f'./Saved_Testing_Data/ppo_never_seen_test_data_rho_{rho}.npz')
    data_all[rho] = d

ref_trajectory = data_all[rhos[0]]['ref_trajectory']

ref_angles = np.array([position_to_angle(x, y, z) for x, y, z in ref_trajectory])
ref_theta = ref_angles[:, 0]  # phi_b (bending angle)
ref_phi = ref_angles[:, 1]    # gamma_r (bending direction)

angles_all = {}
for rho in rhos:
    y_actual = data_all[rho]['y_actual']
    angles = np.array([position_to_angle(x, y, z) for x, y, z in y_actual])
    angles_all[rho] = angles

fig, axes = plt.subplots(5, 1, figsize=(3.5, 4.9), sharex=True,
                         gridspec_kw={'height_ratios': [2, 2, 1, 1, 1]})

angle_labels = ['Bending Angle', 'Bending Direction']
angle_units = ['rad', 'rad']
ref_angle_data = [ref_theta, ref_phi]

for i in range(2):
    ax = axes[i]
    ax.plot(ref_angle_data[i], color='black', label='Reference')
    for rho, color in zip(rhos, colors):
        angle_actual = angles_all[rho][:, i]
        ax.plot(angle_actual, color=color, label=f'ρ={rho}')
    ax.set_ylabel(f'{angle_labels[i]} ({angle_units[i]})')
    if i == 0:
        ax.legend(loc='upper right', borderpad=0.3, labelspacing=0.2,
                  handlelength=1.2, handletextpad=0.4, borderaxespad=0.3)
    ax.grid(True, alpha=0.3)

    # Zoom-in inset around time step 40
    axins = ax.inset_axes([0.02, 0.60, 0.26, 0.36])
    axins.plot(ref_angle_data[i], color='black')
    for rho, color in zip(rhos, colors):
        angle_actual = angles_all[rho][:, i]
        axins.plot(angle_actual, color=color)
    
    axins.set_xlim(79, 89)
    
    if i == 0:
        axins.set_ylim(0.76, 0.80)   
    else:
        axins.set_ylim(1.03, 1.07)   
    axins.set_xticks([])
    axins.set_yticks([])

    axins.grid(True, alpha=0.3)
    ax.indicate_inset_zoom(axins, edgecolor='gray', alpha=0.8)

for j, (rho, color) in enumerate(zip(rhos, colors)):
    ax = axes[2 + j]
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

axes[-1].set_xlabel('Time Steps')

plt.tight_layout()
plt.savefig('./Figure/ppo_angle_tracking_and_trigger_never_seen.png', dpi=300, bbox_inches='tight')
plt.savefig('./Figure/ppo_angle_tracking_and_trigger_never_seen.pdf', bbox_inches='tight')
plt.show()
