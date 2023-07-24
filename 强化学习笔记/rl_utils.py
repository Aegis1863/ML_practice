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

def train_on_policy_agent(env, agent, s_epoch, total_epochs, s_episode, total_episodes, return_list, ckp_path):
    '''
    在线策略, 没有经验池, 仅限演员评论员框架
    '''
    best_score = -1e10  # 初始分数
    if not return_list:
        return_list = []
    for epoch in range(s_epoch, total_epochs):
        with tqdm(total=(total_episodes - s_episode), desc='<%d/%d>' % (epoch + 1, total_epochs), leave=False) as pbar:
            for episode in range(s_episode, total_episodes):
                episode_return = 0
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
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (total_episodes * epoch + episode + 1),
                                      'recent_return': '%.3f' % np.mean(return_list[-10:])})
                    
                if episode_return > best_score:
                    actor_best_weight = agent.actor.state_dict()
                    critic_best_weight = agent.critic.state_dict()
                    best_score = episode_return
                    
                torch.save({
                'epoch': epoch,
                'episode': episode,
                'actor_best_weight': actor_best_weight,
                'critic_best_weight' : critic_best_weight,
                'return_list': return_list,
                }, ckp_path)
                    
                pbar.update(1)
            s_episode = 0
            
    agent.actor.load_state_dict(actor_best_weight)
    agent.critic.load_state_dict(critic_best_weight)
    
    # 如果检查点保存了回报列表, 就无需返回return_list      
    # return return_list

def train_off_policy_agent(env,  agent,  s_epoch, total_epochs, s_episode,  total_episodes,  replay_buffer,
                           minimal_size,  batch_size, return_list, ckp_path):
    '''
    离线策略, 从经验池抽取, 仅限演员评论员框架
    '''
    best_score = 0
    return_list = []
    for epoch in range(s_epoch, total_epochs):
        with tqdm(total=(total_episodes - s_episode), desc='<%d/%d>' % (epoch + 1, total_epochs), leave=False) as pbar:
            for episode in range(s_episode, total_episodes):
                episode_return = 0
                state = env.reset(seed=42)[0]
                done = truncated = False
                while not (done | truncated):
                    action = agent.take_action(state)
                    next_state, reward, done, truncated, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done, truncated)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d, b_t = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d, 'truncated': b_t}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (total_episodes * epoch + episode + 1),
                                      'recent_return': '%.3f' % np.mean(return_list[-10:])})
                    
                if episode_return > best_score:
                    actor_best_weight = agent.actor.state_dict()
                    critic_best_weight = agent.critic.state_dict()
                    best_score = episode_return
                    
                torch.save({
                'epoch': epoch,
                'episode': episode,
                'actor_best_weight': actor_best_weight,
                'critic_best_weight' : critic_best_weight,
                'return_list': return_list,
                }, ckp_path)
                
                pbar.update(1)
            s_episode = 0 
            
    agent.actor.load_state_dict(actor_best_weight)
    agent.critic.load_state_dict(critic_best_weight)
    
    # 如果检查点保存了回报列表, 就无需返回      
    # return return_list

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:  # 逆向折算
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    advantage_list = torch.tensor(np.array(advantage_list), dtype=torch.float)
    # 对advantage_list进行标准化, 因为优势决定了优化方向
    # 有的优势虽然是正的，但是很小，就应该弱化这种优势，标准化后就会变成负的
    # 当然也可以直接输出advantage_list
    advantage_list = (advantage_list - advantage_list.mean())/(advantage_list.std() + 1e-5)
    return advantage_list

def show_gym_policy(env_name, model, model_type: str, render_mode="human", epochs=10, steps=300):
    '''
    `env_name`: 环境名称;\\
    `model`: 类或网络模型, 如agent或agent.net;\\
    `model_type`: 'AC'或'V', 即演员评论员还是价值策略;\\
    `render_mode`: 渲染模式;\\
    `epochs`: 展示轮数\\
    `steps`: 每轮多少步
    '''
    assert model_type in ['V', 'AC'], '模型类别错误, 应输入 V 或 AC'
    env = gym.make(env_name, render_mode=render_mode)
    env.reset()
    totals = []
    for i in range(epochs):  # 测试轮数
        episode_returns = 0
        state = env.reset()[0]  # 第二个输出为info，可以不要
        for _ in range(steps):  # 每回合最多300步
            try:
                if model_type == 'V':
                    Q_values = model(torch.tensor(state).to('cuda'))
                    action = np.argmax(Q_values.tolist())
                    state, reward, done, truncated, info = env.step(action)
                    episode_returns += reward
                elif model_type == 'AC':  # 演员评论员框架的梯度策略
                    action = model.take_action(state)
                    state, reward, done, truncated, info = env.step(action)
                    episode_returns += reward
                else:
                    raise Exception('未识别模型类型')
            except:
                env.close()
                raise Exception('Action execution error!')
            if done or truncated:
                break
        totals.append((episode_returns))
    env.close()
    return np.array(totals).round(3)

def picture_return(return_list, policy_name, env_name, move_avg=9):
    '''传入回报列表, 策略名称, 环境名称, 移动平均周期'''
    sns.set()
    episodes_list = list(range(len(return_list)))
    mv_return = moving_average(return_list, move_avg)
    plt.plot(episodes_list, return_list, label='origin', linestyle='-', alpha=0.5)

    plt.plot(episodes_list, mv_return, label='avg 9', linewidth='1.5')
    plt.title('{} on {}'.format(policy_name, env_name))
    plt.xlabel('Episodes')
    plt.ylabel('Total return')
    plt.legend()
    plt.show()