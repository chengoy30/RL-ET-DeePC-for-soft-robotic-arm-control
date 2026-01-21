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
