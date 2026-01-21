from tqdm import tqdm
import numpy as np
import torch
import collections
import random
import os
import time

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 

    def size(self): 
        return len(self.buffer)

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def train_PPO_agent(env, agent, num_episodes, rho, test_interval=20, save_dir='./saved_models'):
    return_list = []
    action_1_ratio_list = []
    best_test_reward = float('-inf')  # 初始化最佳测试奖励为负无穷
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                global_episode = i * int(num_episodes / 10) + i_episode + 1
                episode_return = 0
                total_action_count = 0  # 初始化动作计数
                action_1_count = 0      # 初始化action=1的计数
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state, _ = env.reset() if i == 0 and i_episode == 0 else env.reset()
                done = False
                env.step(action = 1)
                action_1_count += 1  # initialize the first step action=1
                total_action_count += 1
                while not done:
                    action = agent.take_action(state)
                    if env.k >= env.N - 1:
                        action = 1
                    total_action_count += 1
                    if action == 1:
                        action_1_count += 1
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                action_1_ratio = action_1_count / total_action_count if total_action_count > 0 else 0
                action_1_ratio_list.append(action_1_ratio)
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)

                if global_episode % test_interval == 0:
                    print(f"\n===== 在第 {global_episode} 个episode进行测试 =====")
                    test_obs, test_actions, test_rewards, _, _ = test_PPO_agent(env, agent, num_episodes=1)
                    test_total_reward = sum(test_rewards[0])
                    print(f"测试奖励: {test_total_reward:.3f}, 当前最佳: {best_test_reward:.3f}")
                    
                    if test_total_reward > best_test_reward:
                        best_test_reward = test_total_reward
                        current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
                        best_model_path = os.path.join(save_dir, f"ppo_softarm_{rho}_{current_time}_best.pth")
                        torch.save({
                            'actor': agent.actor.state_dict(),
                            'critic': agent.critic.state_dict()
                        }, best_model_path)
                        print(f"✓ new best model saved! reward: {best_test_reward:.3f}")
                    print("=" * 50)
                
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
    return return_list, action_1_ratio_list, best_test_reward

def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                # gymnasium 返回 (state, info) 元组
                state, _ = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    # gymnasium 返回 5 个值: (next_state, reward, terminated, truncated, info)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    # 先转换为 numpy 数组，避免从 numpy 标量列表创建 tensor 的性能问题
    return torch.tensor(np.array(advantage_list), dtype=torch.float)


def test_DQN_agent(env, agent, num_episodes=1):
    all_observations = []  # 存储所有observation
    all_actions = []       # 存储所有action
    all_rewards = []       # 存储所有reward
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        
        episode_obs = [state.copy()]  # 记录初始状态
        episode_actions = []
        episode_rewards = []
        
        # 初始化步骤（类似训练时）- 也记录这个action=1
        next_state, reward, _ = env.step(action=1)
        episode_actions.append(1)  # 记录初始的action=1
        episode_rewards.append(reward)
        episode_obs.append(next_state.copy())
        state = next_state  # 更新state，与训练时保持一致
        
        while not done:
            # 使用贪婪策略选择action（不使用epsilon探索）
            with torch.no_grad():
                state_tensor = torch.tensor(np.array([state]), dtype=torch.float).to(agent.device)
                action = agent.q_net(state_tensor).argmax().item()
            
            # 与训练时保持一致的约束：k >= N-1 时强制action=1
            if env.k >= env.N - 1:
                action = 1
            
            next_state, reward, done = env.step(action)
            
            episode_obs.append(next_state.copy())
            episode_actions.append(action)
            episode_rewards.append(reward)
            
            state = next_state
        
        all_observations.append(np.array(episode_obs))
        all_actions.append(np.array(episode_actions))
        all_rewards.append(np.array(episode_rewards))
        
        print(f"Episode {ep+1}: Total Reward = {sum(episode_rewards):.3f}, Steps = {len(episode_actions)}")
    
    return all_observations, all_actions, all_rewards

def test_PPO_agent(env, agent, num_episodes=1):
    all_observations = []  # 存储所有observation（误差向量）
    all_actions = []       # 存储所有action
    all_rewards = []       # 存储所有reward
    all_y_actual = []      # 存储实际输出 y
    all_y_target = []      # 存储目标点 target
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        
        episode_obs = [state.copy()]  # 记录初始状态（误差向量）
        episode_actions = []
        episode_rewards = []
        # 额外保存实际位置和目标位置
        episode_y_actual = [env.y.flatten().copy()]  # 初始实际位置
        initial_target = env.y_desired[:, 0]
        episode_y_target = [initial_target.flatten().copy()]  # 初始目标位置
        
        # 初始化步骤（类似训练时）- 也记录这个action=1
        # gymnasium 返回 5 个值: (next_state, reward, terminated, truncated, info)
        next_state, reward, terminated, truncated, _ = env.step(action=1)
        episode_actions.append(1)  # 记录初始的action=1
        episode_rewards.append(reward)
        episode_obs.append(next_state.copy())
        # 保存 step 后的实际位置和目标位置
        episode_y_actual.append(env.y.flatten().copy())
        current_target = env.y_desired[:, min(env.t, env.y_desired.shape[1]-1)]
        episode_y_target.append(current_target.flatten().copy())
        state = next_state  # 更新state，与训练时保持一致
        done = terminated or truncated
        
        while not done:
            # 使用确定性策略选择action（选择概率最高的动作，不使用采样）
            with torch.no_grad():
                state_tensor = torch.tensor(np.array([state]), dtype=torch.float).to(agent.device)
                probs = agent.actor(state_tensor)  # 获取动作概率分布
                action = probs.argmax().item()  # 选择概率最高的动作
            
            # 与训练时保持一致的约束：k >= N-1 时强制action=1
            if env.k >= env.N - 1:
                action = 1
            
            # gymnasium 返回 5 个值: (next_state, reward, terminated, truncated, info)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_obs.append(next_state.copy())
            episode_actions.append(action)
            episode_rewards.append(reward)
            # 保存 step 后的实际位置和目标位置
            episode_y_actual.append(env.y.flatten().copy())
            current_target = env.y_desired[:, min(env.t, env.y_desired.shape[1]-1)]
            episode_y_target.append(current_target.flatten().copy())
            
            state = next_state
        
        all_observations.append(np.array(episode_obs))
        all_actions.append(np.array(episode_actions))
        all_rewards.append(np.array(episode_rewards))
        all_y_actual.append(np.array(episode_y_actual))
        all_y_target.append(np.array(episode_y_target))
        
        print(f"Episode {ep+1}: Total Reward = {sum(episode_rewards):.3f}, Steps = {len(episode_actions)}")
    
    return all_observations, all_actions, all_rewards, all_y_actual, all_y_target