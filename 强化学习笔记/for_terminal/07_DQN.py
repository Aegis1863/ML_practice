#!/usr/bin/env python
# coding: utf-8

# In[1]:


from cProfile import label
import rl_utils
import numpy as np
import random
import gymnasium as gym
import collections
import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os


# # 经验缓存

# In[2]:


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
        
    def add(self, state, action, reward, next_state, done, truncated):
        self.buffer.append((state, action, reward, next_state, done, truncated))
    
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, truncated = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done, truncated  # 多返回了一个state(?)


# # Q网络

# In[3]:


class Qnet(torch.nn.Module):
    def __init__(self, state_dim=4, hidden_dim_1=32, hidden_dim_2=32, action_dim=2):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim_1)
        self.fc2 = torch.nn.Linear(hidden_dim_1, hidden_dim_1)
        self.fc3 = torch.nn.Linear(hidden_dim_2, action_dim)
    
    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        return self.fc3(x)


# # DQN算法

# In[4]:


class DQN:
    ''' DQN算法 '''
    def __init__(self, state_dim, hidden_dim_1, hidden_dim_2, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim_1, hidden_dim_2, self.action_dim).to(device)  # Q网络
        # 目标网络
        self.target_q_net = Qnet(state_dim, hidden_dim_1, hidden_dim_2, self.action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(state[np.newaxis, :], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item() # 转动作序号
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        truncated = torch.tensor(transition_dict['truncated'], dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # 模型预测本状态Q值
        # 👆 tensor.gather(1, actions) 按行取, 索引为动作序号
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1) # 下个状态的最大Q值
        # 👆 tensor.max(1)[0], (1)是指按行取最大,(0)按列; [0]是取值,[1]取序号,等于argmax
        q_targets = rewards + self.gamma * max_next_q_values * (1 - (dones.int() | truncated.int()))  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # 🟠 PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()  # 执行优化

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 更新目标网络
        self.count += 1


# # 初始化

# In[5]:


# 环境相关
env_name = 'CartPole-v1'
env = gym.make(env_name, render_mode='rgb_array')

# DQN相关
total_epoch = 50  # 迭代数
s_epoch = 0 # 默认开始位置
total_episode = 100  # 每次迭代最大游戏轮数
max_step = 500
gamma = 0.98
epsilon = 1  # 刚开始随机动作,更新中线性降低
target_update = 20  # 若干回合更新一次目标网络
buffer_size = 10000  # 经验大小
minimal_size = 1000  # 最小经验数
batch_size = 128
best_score = 0  # 每回合中的最佳分数
replay_buffer = ReplayBuffer(buffer_size)
return_list = []

# 神经网络相关
lr = 2e-3
hidden_dim_1 = 32
hidden_dim_2 = 32
state_dim = env.observation_space.shape[0]  # 状态空间大小
action_dim = env.action_space.n  # 动作空间大小
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
agent = DQN(state_dim, hidden_dim_1, hidden_dim_2, action_dim, lr, gamma, epsilon,
            target_update, device)
best_weight = agent.q_net.state_dict()

# 随机数种子
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# # 读取检查点

# In[6]:


if os.path.exists('checkpoints/ckpt_dqn.pt'):
    checkpoint = torch.load('checkpoints/ckpt_dqn.pt')
    s_epoch = checkpoint['epoch']
    epsilon = checkpoint['epsilon']
    agent.q_net.load_state_dict(checkpoint['best_weight'])
    return_list = checkpoint['return_list']
else:
    s_epoch = 0


# # 迭代算法

# In[25]:

with tqdm.tqdm(total=total_episode) as pbar:
    for epoch in range(s_epoch, total_epoch):
        for episode in range(total_episode):
            episode_return = 0
            state = env.reset()[0]
            done = truncated = False
            step = 0
            while (not (done | truncated)) and step < max_step:  # 执行单次游戏, 最多max_step步
                action = agent.take_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                replay_buffer.add(state, action, reward, next_state, done, truncated)
                state = next_state
                episode_return += reward
                # 当buffer数据的数量超过一定值后,才进行Q网络训练
                if len(replay_buffer.buffer) > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d, b_t = replay_buffer.sample(batch_size)
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d,
                        'truncated': b_t
                    }
                    agent.update(transition_dict)  # 获得经验中抽取的样本, 拟合网络并且梯度上升
                    step += 1 # 步数加一
                    
                    # 获得最佳权重
                    if step > best_score:
                        best_weight = agent.q_net.state_dict()
                        best_score = step
                    
            return_list.append(episode_return)
                
            # 调整epsilon
            agent.epsilon = max(1 - epoch / (total_epoch / 2), 0.01)
            
            # 保存检查点
            torch.save({
                'epoch': epoch,
                'episode': episode,
                'best_weight': best_weight,
                'epsilon': epsilon,
                'return_list': return_list,
                }, 'checkpoints/ckpt_dqn.pt')
            
    pbar.update(1)  # 更新进度条
agent.q_net.load_state_dict(best_weight)  # 应用最佳权重

torch.save(best_weight, 'models/dqn.pth')

# In[10]:


episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list, label='return')
plt.title('DQN on {}'.format(env_name))

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return, label='return of moving_average_9')
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.legend()
plt.savefig('pictures/DQN.jpg')

# %%

print('测试结果: ', rl_utils.show_gym_policy(env_name, agent.q_net, 'rgb_array', epochs=5, steps=500))

env.close()
# %%
