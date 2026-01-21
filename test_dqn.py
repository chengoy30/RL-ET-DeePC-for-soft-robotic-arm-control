import os
import torch
import random
import numpy as np
from Agent.DQN import DQN
from Lib.rl_utils import test_DQN_agent
from SoftArm_env import SoftArmEnv
from Lib.utils import generate_circular_trajectory
from train_dqn import load_data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Lib.SoftArm_lib import SoftArmSection

save_dir = "./saved_models"
model_path = os.path.join(save_dir, "dqn_softarm_0.1_2026-01-21_10-16-09_best.pth")

seed_number = 0
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

lr = 2e-3
hidden_dim = 128
gamma = 0.98
epsilon = 0
target_update = 10
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
            target_update, device)

checkpoint = torch.load(model_path, weights_only=False)
if isinstance(checkpoint, dict) and 'q_net_state_dict' in checkpoint:
    agent.q_net.load_state_dict(checkpoint['q_net_state_dict'])
    agent.target_q_net.load_state_dict(checkpoint['target_q_net_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
else:
    agent.q_net.load_state_dict(checkpoint)
    agent.target_q_net.load_state_dict(checkpoint)  

print("start testing the trained DQN model...")
test_observations, test_actions, test_rewards, test_y_actual, test_y_target = test_DQN_agent(env, agent, num_episodes=1)
print(f"\ntesting completed! total steps: {len(test_actions[0])}")

obs_data = test_observations[0]  
action_data = test_actions[0]     
reward_data = test_rewards[0]     

y_actual = test_y_actual[0]  
y_target = test_y_target[0]  

ref_trajectory = y_desired[:, :len(y_actual)].T  

time_steps = np.arange(len(y_actual))
action_time_steps = np.arange(len(action_data))

fig1, axes1 = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

labels = ['X', 'Y', 'Z']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for i, (ax, label, color) in enumerate(zip(axes1, labels, colors)):
    ax.plot(time_steps, y_actual[:, i], label=f'actual {label}', color=color, linewidth=2)
    ax.plot(time_steps[:len(ref_trajectory)], ref_trajectory[:, i], 
            label=f'reference {label}', color=color, linestyle='--', alpha=0.7, linewidth=2)
    
    ax.set_ylabel(label, fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, len(time_steps)])

axes1[2].set_xlabel('time steps', fontsize=12)
fig1.suptitle('DQN control test - Observation (position) results', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('./Figure/dqn_observation_results.png', dpi=150, bbox_inches='tight')
plt.show()

fig2, axes2 = plt.subplots(2, 1, figsize=(12, 6))

ax_action = axes2[0]
ax_action.scatter(action_time_steps, action_data, color=['#e74c3c' if a == 1 else '#3498db' for a in action_data], 
              alpha=0.7, edgecolor='black', linewidth=0.5)
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
plt.savefig('./Figure/dqn_action_results.png', dpi=150, bbox_inches='tight')
plt.show()

fig3 = plt.figure(figsize=(12, 8))
ax3d = fig3.add_subplot(111, projection='3d')

ax3d.plot(y_actual[:, 0], y_actual[:, 1], y_actual[:, 2], 
          label='real trajectory', color='#e74c3c', linewidth=2)

ax3d.plot(ref_trajectory[:, 0], ref_trajectory[:, 1], ref_trajectory[:, 2], 
          label='reference trajectory', color='#3498db', linestyle='--', linewidth=2, alpha=0.7)

ax3d.set_xlabel('X (mm)', fontsize=11)
ax3d.set_ylabel('Y (mm)', fontsize=11)
ax3d.set_zlabel('Z (mm)', fontsize=11)
ax3d.legend(loc='upper left', fontsize=10)

ax3d.view_init(elev=20, azim=45)

plt.tight_layout()
plt.savefig('./Figure/dqn_trajectory_3d.png', dpi=150, bbox_inches='tight')
plt.show()