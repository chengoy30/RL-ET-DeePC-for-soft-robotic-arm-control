import numpy as np
import matplotlib.pyplot as plt

rhos = [0.1, 0.3, 0.6]
colors = ['#e74c3c', '#3498db', '#2ca02c']
data_all = {}

for rho in rhos:
    d = np.load(f'./Data/ppo_test_data_rho_{rho}.npz')
    data_all[rho] = d

ref_trajectory = data_all[rhos[0]]['ref_trajectory']

fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=True,
                         gridspec_kw={'height_ratios': [2, 2, 2, 1, 1, 1]})

# --- Top 3 subplots: tracking trajectory for X, Y, Z ---
axis_labels = ['X', 'Y', 'Z']
for i in range(3):
    ax = axes[i]
    ax.plot(ref_trajectory[:, i], color='black', linewidth=2, label='Reference')
    for rho, color in zip(rhos, colors):
        y_actual = data_all[rho]['y_actual']
        ax.plot(y_actual[:, i], color=color, linewidth=1.5, label=f'ρ={rho}')
    ax.set_ylabel(f'{axis_labels[i]} (mm)', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

# --- Bottom 3 subplots: trigger (action) for each rho ---
for j, (rho, color) in enumerate(zip(rhos, colors)):
    ax = axes[3 + j]
    action_data = data_all[rho]['action_data']
    time_steps = np.arange(len(action_data))
    ax.bar(time_steps, action_data, color=color, alpha=0.8, width=0.5)
    ax.set_ylabel('Trigger', fontsize=11)
    ax.set_yticks([0, 1])
    ax.set_ylim(-0.1, 1.3)
    ax.text(0.98, 0.85, f'ρ={rho}', transform=ax.transAxes,
            fontsize=10, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.2))
    ax.grid(True, alpha=0.3, axis='y')

axes[-1].set_xlabel('Step, k', fontsize=12)

plt.tight_layout()
plt.savefig('./Figure/ppo_tracking_error_and_trigger.png', dpi=150, bbox_inches='tight')
plt.show()
