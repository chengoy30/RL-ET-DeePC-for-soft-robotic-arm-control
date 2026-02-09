import os
import torch
import random
import numpy as np
from Agent.A2C import A2C
from SoftArm_env import SoftArmEnv
from Lib.SoftArm_lib import SoftArmSection
from Lib.utils import generate_circular_trajectory
from train_a2c import load_data
import matplotlib.pyplot as plt
from Lib.rl_utils import test_A2C_agent
from mpl_toolkits.mplot3d import Axes3D

save_dir = "./Saved_Models"
model_path = os.path.join(save_dir, "a2c_softarm_0.1_2026-02-06_17-34-03_best.pth")  

seed_number = 10
random.seed(seed_number)
np.random.seed(seed_number)
torch.manual_seed(seed_number)

rho = 0.1

param_deepc = load_data()
Tini = param_deepc[4]
N = param_deepc[5]

arm_section = SoftArmSection(
    n=3,
    L=9.30,
    d=1.25,
    Kb=20.02,
    Kc=3.10
)

total_steps = 200
circular_trajectory = generate_circular_trajectory(
    num_points=144,
    theta=np.pi/4,
    L_arm=93.0,
    total_steps=total_steps,
    N=N
)

y_desired = circular_trajectory.T

env = SoftArmEnv(param_deepc, arm_section, y_desired, rho)
state, _ = env.reset(seed=seed_number)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

actor_lr = 1e-3
critic_lr = 1e-2
hidden_dim = 128
gamma = 0.98
lmbda = 0.95
entropy_coef = 0.01
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

agent = A2C(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, gamma, entropy_coef, device)

agent.load_model(model_path)

print("\n===== start testing =====")
test_observations, test_actions, test_rewards, test_y_actual, test_y_target = test_A2C_agent(env, agent)

obs_data = test_observations[0]
action_data = test_actions[0]
reward_data = test_rewards[0]

y_actual = test_y_actual[0]
y_target = test_y_target[0]

ref_trajectory = y_desired[:, :len(y_actual)].T

np.savez(f'./Saved_Testing_Data/a2c_test_data_rho_{rho}.npz',
         obs_data=obs_data, action_data=action_data, reward_data=reward_data,
         y_actual=y_actual, y_target=y_target, ref_trajectory=ref_trajectory)

# calculate the tracking error MSE (Mean Squared Error of L2 norm)
# MSE = (1/n) * Σ ||y_actual - y_ref||²
err = y_actual - ref_trajectory
mse_total = np.mean(np.sum(err**2, axis=1))
rmse_total = np.sqrt(mse_total)

print("\n" + "="*50)
print("Tracking Error Statistics")
print("="*50)
print(f"MSE: {mse_total:.6f} mm²")
print("-"*50)
print(f"RMSE: {rmse_total:.4f} mm")
print("="*50 + "\n")

time_steps = np.arange(len(y_actual))
action_time_steps = np.arange(len(action_data))

fig1, axes1 = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

labels = ['X', 'Y', 'Z']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for i, (ax, label, color) in enumerate(zip(axes1, labels, colors)):
    ax.plot(time_steps, y_actual[:, i], label=f'real {label}', color=color, linewidth=2)
    ax.plot(time_steps[:len(ref_trajectory)], ref_trajectory[:, i],
            label=f'reference {label}', color=color, linestyle='--', alpha=0.7, linewidth=2)

    ax.set_ylabel(label, fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, len(time_steps)])

    if i == 2:
        ax.set_ylim(-85, -81)

axes1[2].set_xlabel('time step', fontsize=12)
fig1.suptitle('A2C test - Observation (position) results', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'./Figure/a2c_observation_results_rho_{rho}.png', dpi=150, bbox_inches='tight')
plt.show()

fig2, axes2 = plt.subplots(2, 1, figsize=(12, 6))

ax_action = axes2[0]
ax_action.bar(action_time_steps, action_data, color=['#e74c3c' if a == 1 else '#3498db' for a in action_data],
              alpha=0.7, edgecolor='black', linewidth=0.5, width=0.3)
ax_action.set_ylabel('Action', fontsize=12)
ax_action.set_xlabel('Time Step', fontsize=12)
ax_action.set_yticks([0, 1])
ax_action.set_yticklabels(['0', '1'])
ax_action.grid(True, alpha=0.3, axis='y')
ax_action.set_xlim([-1, len(action_time_steps)])

ax_pie = axes2[1]
action_counts = [np.sum(action_data == 0), np.sum(action_data == 1)]
labels_pie = [f'Action 0\n{action_counts[0]}times',
              f'Action 1\n{action_counts[1]}times']
colors_pie = ['#3498db', '#e74c3c']
explode = (0.05, 0.05)

wedges, texts, autotexts = ax_pie.pie(action_counts, explode=explode, labels=labels_pie,
                                       colors=colors_pie, autopct='%1.1f%%',
                                       shadow=True, startangle=90)

plt.tight_layout()
plt.savefig(f'./Figure/a2c_action_results_rho_{rho}.png', dpi=150, bbox_inches='tight')
plt.show()

fig3, (ax3d, ax2d) = plt.subplots(1, 2, figsize=(18, 8),
                                   subplot_kw={'projection': '3d'})

ax3d.plot(y_actual[-145:, 0], y_actual[-145:, 1], y_actual[-145:, 2],
          label='real trajectory', color='#e74c3c', linewidth=2)
ax3d.plot(ref_trajectory[:, 0], ref_trajectory[:, 1], ref_trajectory[:, 2],
          label='reference trajectory', color='#3498db', linestyle='--', linewidth=2, alpha=0.7)
ax3d.set_xlabel('X (mm)', fontsize=11)
ax3d.set_ylabel('Y (mm)', fontsize=11)
ax3d.set_zlabel('Z (mm)', fontsize=11)
ax3d.legend(loc='upper left', fontsize=10)
ax3d.set_zlim(-95, -75)
ax3d.view_init(elev=20, azim=225)
ax3d.set_title('3D View', fontsize=13)

ax2d.remove()
ax2d = fig3.add_subplot(122)

ax2d.plot(y_actual[-145:, 0], y_actual[-145:, 1],
          label='real trajectory', color='#e74c3c', linewidth=2)
ax2d.plot(ref_trajectory[:, 0], ref_trajectory[:, 1],
          label='reference trajectory', color='#3498db', linestyle='--', linewidth=2, alpha=0.7)
ax2d.set_xlabel('X (mm)', fontsize=11)
ax2d.set_ylabel('Y (mm)', fontsize=11)
ax2d.legend(loc='upper left', fontsize=10)
ax2d.set_title('Top View (X-Y)', fontsize=13)
ax2d.set_aspect('equal')
ax2d.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'./Figure/a2c_trajectory_3d_rho_{rho}.png', dpi=150, bbox_inches='tight')
plt.show()
