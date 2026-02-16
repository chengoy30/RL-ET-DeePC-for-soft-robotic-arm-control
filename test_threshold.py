import random
import numpy as np
import torch
from Lib.SoftArm_lib import SoftArmSection
from SoftArm_env_test import TimedSoftArmEnv
from Lib.utils import generate_circular_trajectory
import matplotlib.pyplot as plt

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
    seed_number = 10
    random.seed(seed_number)
    np.random.seed(seed_number)
    torch.manual_seed(seed_number)

    rho = 1.0
    threshold = 0.30

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

    env = TimedSoftArmEnv(param_deepc, arm_section, y_desired, rho)
    state, _ = env.reset(seed=seed_number)

    actions = []
    rewards = []
    y_actual = []
    y_target = []

    print("start testing the threshold control strategy...")

    action = 1
    next_state, reward, terminated, truncated, _ = env.step(action)
    actions.append(action)
    rewards.append(reward)
    y_actual.append(env.y.flatten().copy())
    y_target.append(env.y_desired[:, min(env.t-1, env.y_desired.shape[1]-1)].flatten().copy())
    state = next_state
    done = terminated or truncated

    while not done:
        current_y = y_actual[-1]
        current_target = y_target[-1]
        distance = np.linalg.norm(current_y - current_target)

        action = 1 if distance > threshold else 0
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        actions.append(action)
        rewards.append(reward)
        y_actual.append(env.y.flatten().copy())
        y_target.append(env.y_desired[:, min(env.t-1, env.y_desired.shape[1]-1)].flatten().copy())
        state = next_state

    action_data = np.array(actions)
    reward_data = np.array(rewards)
    y_actual = np.array(y_actual)
    y_target = np.array(y_target)

    print(f"total reward: {sum(reward_data):.4f}")

    print(f"\nTotal DeePC decision time: {env.total_deepc_time:.4f} seconds")
    print(f"DeePC solve calls: {env.deepc_call_count} / {len(env.deepc_times)} steps")
    print(f"Average DeePC solve time: {env.total_deepc_time / max(env.deepc_call_count, 1):.4f} seconds")
    print(f"Trigger rate: {env.deepc_call_count / len(env.deepc_times) * 100:.1f}%")

    ref_trajectory = y_desired[:, :len(y_actual)].T

    # calculate the tracking error MSE (Mean Squared Error of L2 norm)
    # MSE = (1/n) * Σ ||y_actual - y_ref||² (last 144 steps only)
    err = y_actual[-144:] - ref_trajectory[-144:]
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
        ax.plot(time_steps, y_actual[:, i], label=f'actual {label}', color=color, linewidth=2)
        ax.plot(time_steps[:len(ref_trajectory)], ref_trajectory[:, i], 
                label=f'reference {label}', color=color, linestyle='--', alpha=0.7, linewidth=2)
        
        ax.set_ylabel(label, fontsize=12)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, len(time_steps)])
        if i == 2:
            ax.set_ylim(-85, -81)

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

    ax3d.set_zlim(-85, -81)

    ax3d.set_xlabel('X (mm)', fontsize=11)
    ax3d.set_ylabel('Y (mm)', fontsize=11)
    ax3d.set_zlabel('Z (mm)', fontsize=11)
    ax3d.legend(loc='upper left', fontsize=10)

    ax3d.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.savefig('./Figure/dqn_trajectory_3d.png', dpi=150, bbox_inches='tight')
    plt.show()