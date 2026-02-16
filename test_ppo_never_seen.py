import os
import torch
import random
import numpy as np
from Agent.PPO import PPO
from SoftArm_env import SoftArmEnv
from Lib.SoftArm_lib import SoftArmSection
from train_ppo import load_data
import matplotlib.pyplot as plt
from Lib.rl_utils import test_PPO_agent
from Lib.SoftArm_lib import constant_curvature

save_dir = "./Saved_Models"
# model_path = os.path.join(save_dir, "ppo_softarm_0.1_2026-02-15_12-13-35_best.pth")
# model_path = os.path.join(save_dir, "ppo_softarm_0.5_2026-02-15_13-23-56_best.pth")
model_path = os.path.join(save_dir, "ppo_softarm_1.0_2026-02-15_14-52-17_best.pth")

def load_data():
    data = np.load("./Data/hankel_matrices.npz", allow_pickle=True)
    Up = data["Up"]
    Uf = data["Uf"]
    Yp = data["Yp"]
    Yf = data["Yf"]
    Tini = int(data["Tini"].item())
    N = int(data["N"].item())
    p_ctr = int(data["p"].item())
    m_ctr = int(data["m"].item())
    Q = 800.0 * np.eye(p_ctr)
    R = 1e-5 * np.eye(m_ctr)
    lambda_g = 300
    lambda_y = 1500
    u_limit = np.array([[-1200, 1200],
                        [-1200, 1200],
                        [-1200, 1200]], dtype=float)
    y_limit = np.array([[-100, 100],
                        [-100, 100],
                        [-100, 0]], dtype=float)
    param_deepc = [Up, Yp, Uf, Yf, Tini, N, Q, R, lambda_g, lambda_y, u_limit, y_limit]
    return param_deepc

if __name__ == "__main__":
    seed_number = 40
    random.seed(seed_number)
    np.random.seed(seed_number)
    torch.manual_seed(seed_number)

    rho = 1.0

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
    trajectory = np.zeros((total_steps, 3), dtype=float)

    for i in range(total_steps):
        if i <= 40:
            theta = np.deg2rad(15.0)
            phi   = np.deg2rad(20.0)
        elif i <= 80:
            theta = np.deg2rad(30.0)
            phi   = np.deg2rad(40.0)
        elif i <= 120:
            theta = np.deg2rad(45.0)
            phi   = np.deg2rad(60.0)
        elif i <= 160:
            theta = np.deg2rad(30.0)
            phi   = np.deg2rad(40.0)
        elif i <= 200:
            theta = np.deg2rad(15.0)
            phi   = np.deg2rad(20.0)
        

        x_r, y_r, z_r = constant_curvature(theta, phi, 93.0)
        trajectory[i, :] = [x_r, y_r, -z_r]

    y_desired = trajectory.T
    env = SoftArmEnv(param_deepc, arm_section, y_desired, rho)
    state, _ = env.reset(seed=seed_number)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 100
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                    epochs, eps, gamma, device)

    agent.load_model(model_path)

    print("\n===== start testing =====") 
    test_observations, test_actions, test_rewards, test_y_actual, test_y_target = test_PPO_agent(env, agent)

    obs_data = test_observations[0]
    action_data = test_actions[0]
    reward_data = test_rewards[0]

    y_actual = test_y_actual[0]
    y_target = test_y_target[0]

    ref_trajectory = y_desired[:, :len(y_actual)].T

    np.savez(f'./Saved_Testing_Data/ppo_never_seen_test_data_rho_{rho}.npz',
         obs_data=obs_data, action_data=action_data, reward_data=reward_data,
         y_actual=y_actual, y_target=y_target, ref_trajectory=ref_trajectory)

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

    axes1[2].set_xlabel('time step', fontsize=12)
    fig1.suptitle('PPO test - Observation (position) results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'./Figure/ppo_never_seen_observation_results_rho_{rho}.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("observation results saved to: ppo_never_seen_observation_results.png")

    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 6))

    ax_action = axes2[0]
    ax_action.bar(action_time_steps, action_data, color=['#e74c3c' if a == 1 else '#3498db' for a in action_data], 
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
    plt.savefig(f'./Figure/ppo_never_seen_action_results_rho_{rho}.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("action results saved to: ppo_never_seen_action_results.png")