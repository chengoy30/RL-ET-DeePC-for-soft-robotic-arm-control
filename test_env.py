import random
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import Lib.rl_utils as rl_utils
from Lib.SoftArm_lib import SoftArmSection
from SoftArm_env import SoftArmEnv
import os
from Lib.utils import generate_circular_trajectory
import time
from Agent.PPO import PPO

save_dir = "./saved_models"
os.makedirs(save_dir, exist_ok=True)

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

    env = SoftArmEnv(param_deepc, arm_section, y_desired, rho=0.01)
    state, _ = env.reset(seed=seed_number)

    actor_lr = 1e-3
    critic_lr = 1e-2
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    epochs = 10 
    eps = 0.2
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)

    print("start testing the environment...")
    time_start = time.time()
    state, _ = env.reset(seed=seed_number)
    
    rewards_list = []
    y_list = []
    y_desired_list = []
    
    done = False
    step_count = 0
    
    while not done:
        action = 1
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        rewards_list.append(reward)
        y_list.append(env.y.flatten().copy())
        current_target = env.y_desired[:, min(env.t, env.y_desired.shape[1]-1)]
        y_desired_list.append(current_target.flatten().copy())
        
        step_count += 1
        if step_count % 20 == 0:
            print(f"Step {step_count}: reward = {reward:.4f}, y = {env.y.flatten()}")
        
        state = next_state
    
    print(f"testing completed! total steps: {step_count}, total reward: {sum(rewards_list):.4f}")
    time_end = time.time()
    print(f"testing time: {time_end - time_start:.2f} seconds")
    
    rewards_array = np.array(rewards_list)
    y_array = np.array(y_list)
    y_desired_array = np.array(y_desired_list)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(rewards_array, 'b-', linewidth=1)
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Reward over Time')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(y_array[:, 0], 'b-', label='Actual X', linewidth=1)
    axes[0, 1].plot(y_desired_array[:, 0], 'r--', label='Desired X', linewidth=1)
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('X (cm)')
    axes[0, 1].set_title('X Position Tracking')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(y_array[:, 1], 'b-', label='Actual Y', linewidth=1)
    axes[1, 0].plot(y_desired_array[:, 1], 'r--', label='Desired Y', linewidth=1)
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Y (cm)')
    axes[1, 0].set_title('Y Position Tracking')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(y_array[:, 2], 'b-', label='Actual Z', linewidth=1)
    axes[1, 1].plot(y_desired_array[:, 2], 'r--', label='Desired Z', linewidth=1)
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Z (cm)')
    axes[1, 1].set_title('Z Position Tracking')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('./Figure/env_test_result.png', dpi=150)
    plt.show()
    
    print("image saved to ./Figure/env_test_result.png")
    
    fig_3d = plt.figure(figsize=(10, 8))
    ax = fig_3d.add_subplot(111, projection='3d')
    
    ax.plot(y_array[:, 0], y_array[:, 1], y_array[:, 2], 
            'b-', linewidth=1.5, label='Actual Trajectory')
    
    ax.plot(y_desired_array[:, 0], y_desired_array[:, 1], y_desired_array[:, 2], 
            'r--', linewidth=1.5, label='Desired Trajectory')
    
    ax.scatter(y_array[0, 0], y_array[0, 1], y_array[0, 2], 
               c='green', s=100, marker='o', label='Start (Actual)')
    ax.scatter(y_array[-1, 0], y_array[-1, 1], y_array[-1, 2], 
               c='blue', s=100, marker='s', label='End (Actual)')
    ax.scatter(y_desired_array[0, 0], y_desired_array[0, 1], y_desired_array[0, 2], 
               c='orange', s=100, marker='^', label='Start (Desired)')
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('3D Trajectory Tracking')
    ax.legend(loc='upper left')

    ax.set_zlim(-85, -81)
    
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    plt.savefig('./Figure/env_test_result_3d.png', dpi=150)
    plt.show()
    
    print("3D image saved to ./Figure/env_test_result_3d.png")