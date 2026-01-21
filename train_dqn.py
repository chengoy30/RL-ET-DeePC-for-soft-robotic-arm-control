import random
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import Lib.rl_utils as rl_utils
from Agent.DQN import DQN, ReplayBuffer
from Lib.rl_utils import test_DQN_agent
from SoftArm_env import SoftArmEnv
from Lib.SoftArm_lib import SoftArmSection
import os
from Lib.utils import generate_circular_trajectory
import time

save_dir = "./Saved_Models"
save_dir_training_data = "./Saved_Training_Data"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(save_dir_training_data, exist_ok=True)

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
    seed_number = 0
    random.seed(seed_number)
    np.random.seed(seed_number)
    torch.manual_seed(seed_number)

    rho = 0.1
    num_episodes = 10
    test_interval = 5 

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
    epsilon = 0.02
    target_update = 10
    buffer_size = 5000
    minimal_size = 500
    batch_size = 64
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
                target_update, device)

    return_list = []
    action_1_ratio_list = [] 
    best_test_reward = float('-inf') 

    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                global_episode = i * int(num_episodes / 10) + i_episode + 1
                
                episode_return = 0
                action_1_count = 0  
                total_action_count = 0  
                state, _ = env.reset()
                done = False
                
                next_state, reward, terminated, truncated, _ = env.step(action=1)
                state = next_state
                episode_return += reward
                action_1_count += 1
                total_action_count += 1
                done = terminated or truncated
                
                while not done:
                    action = agent.take_action(state)
                    if env.k >= env.N - 1:
                        action = 1
                    total_action_count += 1
                    if action == 1:
                        action_1_count += 1
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    replay_buffer.add(state, action, reward, next_state, done)  
                    state = next_state
                    episode_return += reward
                    # Start training Q-network only when buffer has enough data
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent.update(transition_dict)
                return_list.append(episode_return)
                action_1_ratio = action_1_count / total_action_count if total_action_count > 0 else 0
                action_1_ratio_list.append(action_1_ratio)
                pbar.update(1)

                if global_episode % test_interval == 0:
                    tqdm.write(f"\n===== testing in the {global_episode}th episode =====")
                    test_obs, test_actions, test_rewards, _, _ = test_DQN_agent(env, agent, num_episodes=1)
                    test_total_reward = sum(test_rewards[0])
                    tqdm.write(f"test reward: {test_total_reward:.3f}, best reward: {best_test_reward:.3f}")
                    
                    if test_total_reward > best_test_reward:
                        best_test_reward = test_total_reward
                        current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
                        best_model_path = os.path.join(save_dir, f"dqn_softarm_{rho}_{current_time}_best.pth")
                        torch.save(agent.q_net.state_dict(), best_model_path)
                        tqdm.write(f"new best model saved! reward: {best_test_reward:.3f}")
                    tqdm.write("=" * 50)
                
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on SoftArmEnv')
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on SoftArmEnv')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(episodes_list, action_1_ratio_list, label='action=1 ratio', color='orange')
    plt.xlabel('Episodes')
    plt.ylabel('Action=1 Ratio')
    plt.title('Action=1 ratio in each episode')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    mv_action_1_ratio = rl_utils.moving_average(action_1_ratio_list, 9)
    plt.figure(figsize=(10, 6))
    plt.plot(episodes_list, mv_action_1_ratio, label='action=1 ratio (moving average)', color='green')
    plt.xlabel('Episodes')
    plt.ylabel('Action=1 Ratio (Moving Average)')
    plt.title('Action=1 ratio in each episode (moving average)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    training_data_path = os.path.join(save_dir_training_data, f"dqn_training_data_rho_{rho}_{current_time}.npz")
    np.savez(training_data_path,
             return_list=np.array(return_list),
             mv_return=np.array(mv_return),
             action_1_ratio_list=np.array(action_1_ratio_list),
             mv_action_1_ratio=np.array(mv_action_1_ratio),
             episodes_list=np.array(episodes_list),
             rho=rho,
             num_episodes=num_episodes,
             best_test_reward=best_test_reward,
             seed_number=seed_number)
    print(f"\n training data saved to: {training_data_path}")
