import numpy as np
import torch
import collections
import random
import torch.nn.functional as F

class ReplayBuffer:
    ''' Experience Replay Buffer '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # Queue with FIFO

    def add(self, state, action, reward, next_state, done):  # Add data to buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # Sample data from buffer with batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # Return current buffer size
        return len(self.buffer)
    
class Qnet(torch.nn.Module):
    ''' Q-Network with one hidden layer '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Hidden layer uses ReLU activation function
        return self.fc2(x)
    
class DQN:
    ''' DQN Algorithm '''
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim,
                          self.action_dim).to(device)  # Q-network
        # Target network
        self.target_q_net = Qnet(state_dim, hidden_dim,
                                 self.action_dim).to(device)
        # Use Adam optimizer
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Epsilon-greedy policy
        self.target_update = target_update  # Target network update frequency
        self.count = 0  # Counter to record number of updates
        self.device = device

    def take_action(self, state):  # Take action using epsilon-greedy policy
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q values
        # Max Q value of next state
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(
            -1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones
                                                                )  # TD target
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # MSE loss function
        self.optimizer.zero_grad()  # Clear gradients (PyTorch accumulates gradients by default)
        dqn_loss.backward()  # Backpropagation to update parameters
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # Update target network
        self.count += 1

