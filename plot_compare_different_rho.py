import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# IEEE double-column width: 7.16 in
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

rhos = [0.1, 0.5, 1.0]
colors = ['#e74c3c', '#3498db', '#2ca02c']
data_all = {}

for rho in rhos:
    d = np.load(f'./Saved_Testing_Data/ppo_test_data_rho_{rho}.npz')
    data_all[rho] = d

ref_trajectory = data_all[rhos[0]]['ref_trajectory']

fig, axes = plt.subplots(4, 1, figsize=(3.5, 3.5), sharex=True,
                         gridspec_kw={'height_ratios': [2, 1, 1, 1]})

# --- Top subplot: per-step tracking RMSE for each rho ---
ax = axes[0]
for rho, color in zip(rhos, colors):
    y_actual = data_all[rho]['y_actual']
    rmse_per_step = np.sqrt(np.mean((y_actual - ref_trajectory) ** 2, axis=1))
    ax.plot(rmse_per_step, color=color, label=f'ρ={rho}')
ax.set_ylim(0, 1)
ax.set_ylabel('Tracking Error (mm)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# --- Bottom 3 subplots: trigger (action) for each rho ---
for j, (rho, color) in enumerate(zip(rhos, colors)):
    ax = axes[1 + j]
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

axes[-1].set_xlabel('Time Step')

plt.tight_layout()
plt.savefig('./Figure/ppo_tracking_error_and_trigger.png', dpi=300, bbox_inches='tight')
plt.savefig('./Figure/ppo_tracking_error_and_trigger.pdf', bbox_inches='tight')
plt.show()
