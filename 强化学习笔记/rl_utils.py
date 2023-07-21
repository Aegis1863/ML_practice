from tqdm import tqdm
import numpy as np
import torch
import collections
import random
import gymnasium as gym
import seaborn as sns
import matplotlib.pyplot as plt

class ReplayBuffer:
    '''经验缓存
    
    输入一个整数
    '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done, truncated): 
        self.buffer.append((state, action, reward, next_state, done, truncated)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, truncated = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done, truncated

    def size(self): 
        return len(self.buffer)

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def train_on_policy_agent(env, agent, s_epoch, total_epoch, s_episode, total_episode, reward_list, ckp_path):
    '''
    在线策略, 没有经验池
    '''
    best_score = 0
    if not reward_list:
        reward_list = []
    for epoch in range(s_epoch, total_epoch):
        with tqdm(total=(total_episode - s_episode), desc='<%d/%d>' % (epoch + 1, total_epoch), leave=False) as pbar:
            for episode in range(s_episode, total_episode):
                episode_reward = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [],
                                   'dones': [], 'truncated': []}
                state = env.reset()[0]
                done = truncated = False
                while not (done | truncated):
                    action = agent.take_action(state)
                    next_state, reward, done, truncated, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    transition_dict['truncated'].append(truncated)
                    state = next_state
                    episode_reward += reward
                reward_list.append(episode_reward)
                agent.update(transition_dict)
                if (episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (total_episode * epoch + episode + 1),
                                      'recent_return': '%.3f' % np.mean(reward_list[-10:])})
                    
                if episode_reward > best_score:
                    actor_best_weight = agent.actor.state_dict()
                    critic_best_weight = agent.critic.state_dict()
                    best_score = episode_reward
                    
                torch.save({
                'epoch': epoch,
                'episode': episode,
                'actor_best_weight': actor_best_weight,
                'critic_best_weight' : critic_best_weight,
                'reward_list': reward_list,
                }, ckp_path)
                    
                pbar.update(1)
            s_episode = 0
            
    agent.actor.load_state_dict(actor_best_weight)
    agent.critic.load_state_dict(critic_best_weight)
            
    return reward_list

def train_off_policy_agent(env,  agent,  s_epoch, total_epoch, s_episode,  total_episode,  replay_buffer,
                           minimal_size,  batch_size, reward_list, ckp_path):
    '''
    离线策略, 从经验池抽取
    '''
    best_score = 0
    reward_list = []
    for epoch in range(s_epoch, total_epoch):
        with tqdm(total=(total_episode - s_episode), desc='<%d/%d>' % (epoch + 1, total_epoch), leave=False) as pbar:
            for episode in range(s_episode, total_episode):
                episode_reward = 0
                state = env.reset()[0]
                done = truncated = False
                while not (done | truncated):
                    action = agent.take_action(state)
                    next_state, reward, done, truncated, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done, truncated)
                    state = next_state
                    episode_reward += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d, b_t = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d, 'truncated': b_t}
                        agent.update(transition_dict)
                reward_list.append(episode_reward)
                if (episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (total_episode * epoch + episode + 1),
                                      'recent_return': '%.3f' % np.mean(reward_list[-10:])})
                    
                if episode_reward > best_score:
                    actor_best_weight = agent.actor.state_dict()
                    critic_best_weight = agent.critic.state_dict()
                    best_score = episode_reward
                    
                torch.save({
                'epoch': epoch,
                'episode': episode,
                'actor_best_weight': actor_best_weight,
                'critic_best_weight' : critic_best_weight,
                'reward_list': reward_list,
                }, ckp_path)
                
                pbar.update(1)
            s_episode = 0 
            
    agent.actor.load_state_dict(actor_best_weight)
    agent.critic.load_state_dict(critic_best_weight)
        
    return reward_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)

def show_gym_policy(name, model, render_mode="human", epochs=10, steps=300):
    env = gym.make(name, render_mode=render_mode)
    env.reset()
    totals = []
    for i in range(epochs):  # 测试轮数
        episode_rewards = 0
        obs = env.reset()[0]  # 第二个输出为info，可以不要
        for _ in range(steps):  # 每回合最多300步
            try:
                Q_values = model(torch.tensor(obs).to('cuda'))
                action = np.argmax(Q_values.tolist())
                obs, reward, done, truncated, info = env.step(action)
                episode_rewards += reward
            except:
                env.close()
                raise Exception('Action execution error!')
            if done or truncated:
                break
        totals.append((episode_rewards))
    env.close()
    return totals

def picture_reward(reward_list, policy_name, env_name, move_avg=9):
    '''传入回报列表, 策略名称, 环境名称, 移动平均周期'''
    sns.set()
    episodes_list = list(range(len(reward_list)))
    mv_return = moving_average(reward_list, move_avg)
    plt.plot(episodes_list, reward_list, label='origin', linestyle='-', alpha=0.5)

    plt.plot(episodes_list, mv_return, label='avg 9', linewidth='1.5')
    plt.title('{} on {}'.format(policy_name, env_name))
    plt.xlabel('Episodes')
    plt.ylabel('Total reward')
    plt.legend()
    plt.show()