from tqdm import tqdm
import numpy as np
import torch
import collections
import random
import gymnasium as gym
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import time

class ReplayBuffer:
    '''经验缓存

    输入一个整数
    '''

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state:np.ndarray, action:torch.tensor, reward:float, next_state:np.ndarray, done:int, truncated:int):
        self.buffer.append((state, action, reward, next_state, done, truncated))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, truncated = zip(*transitions)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), done, truncated

    def size(self):
        return len(self.buffer)


def picture_return(return_list, policy_name, env_name, move_avg=10):
    '''生成图像

    参数 Parameters
    ----------
    - return_list : list
        需要画图的数据
    - policy_name : str
        算法名称
    - env_name : str
        环境名称
    - move_avg : int, 可选
        移动周期, 若写0则不显示移动平均, 默认 10
    '''
    sns.set()
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    return_list = pd.Series(return_list)
    alpha = 0.5 if move_avg else 1
    return_list.plot(alpha=alpha, label='原数据')
    if move_avg:
        return_list.rolling(window=move_avg).mean().plot(label='%d次移动平均' % move_avg, linewidth=1.5)
    plt.title('%s on %s' % (policy_name, env_name), fontsize=13)
    plt.xlabel('训练轮数', fontsize=13)
    plt.ylabel('总回报', fontsize=13)
    plt.legend(fontsize=13)
    plt.show()


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] -
              cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def train_on_policy_agent(env, agent, s_epoch, total_epochs, s_episode, total_episodes, return_list, ckp_path):
    '''
    在线策略, 没有经验池, 仅限演员评论员框架
    '''
    start_time = time.time()
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
                agent.update(transition_dict)  # 更新参数
                
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
                    'critic_best_weight': critic_best_weight,
                    'return_list': return_list,
                }, ckp_path)

                pbar.update(1)
            s_episode = 0
    try:        
        agent.actor.load_state_dict(actor_best_weight)
        agent.critic.load_state_dict(critic_best_weight)
    except:
        raise 'please check file route of checkpoints...'
    end_time = time.time()
    print('总耗时: %i分钟' % ((end_time - start_time)/60))
    # 如果检查点保存了回报列表, 可以不返回return_list
    return return_list


def train_off_policy_agent(env, agent, s_epoch, total_epochs, s_episode, total_episodes, replay_buffer, 
                           minimal_size, batch_size, return_list, ckp_path, net_num=2):
    '''离线策略, 从经验池抽取, 仅限演员评论员框架

    参数 Parameters
    ----------
    ckp_path : str
        检查点路径
    net_num : int, 可选
        网络个数, 因部分网络有两个评论员, 共三个网络 ; by default 2

    返回 Returns
    -------
    return_list
        默认保存在检查点, 因此无返回
    '''

    assert net_num >= 2, 'AC网络至少要有两个网络...'
    assert (s_epoch <= total_epochs) or (s_episode <= total_episodes), '结束轮数小于起始轮数...'
    start_time = time.time()
    if not return_list:
        return_list = []
    best_score = -1e10  # 初始分数
    for epoch in range(s_epoch, total_epochs):
        with tqdm(total=(total_episodes - s_episode), desc='<%d/%d>' % (epoch + 1, total_epochs), leave=False) as pbar:
            for episode in range(s_episode, total_episodes):
                episode_return = 0
                state = env.reset()[0]
                done = truncated = False
                while not (done | truncated):
                    action = agent.take_action(state)
                    next_state, reward, done, truncated, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done, truncated)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:  # 确保先收集到一定量的数据再采样
                        b_s, b_a, b_r, b_ns, b_d, b_t = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 
                                           'rewards': b_r, 'dones': b_d, 'truncated': b_t}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if episode_return > best_score:
                    actor_best_weight = agent.actor.state_dict()
                    if net_num == 3:
                        critic_1_best_weight = agent.critic_1.state_dict()
                        critic_2_best_weight = agent.critic_2.state_dict()
                    else:
                        critic_best_weight = agent.critic.state_dict()
                    best_score = episode_return
                    
                if net_num == 3:
                    torch.save({
                        'epoch': epoch,
                        'episode': episode,
                        'actor_best_weight': actor_best_weight,
                        'critic_1_best_weight': critic_1_best_weight,
                        'critic_2_best_weight': critic_2_best_weight,
                        'return_list': return_list,
                    }, ckp_path)
                else:
                    torch.save({
                        'epoch': epoch,
                        'episode': episode,
                        'actor_best_weight': actor_best_weight,
                        'critic_1_best_weight': critic_best_weight, 
                        'return_list': return_list,
                        }, ckp_path)
                    
                if (episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (total_episodes * epoch + episode + 1),
                                      'recent_return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
            s_episode = 0
    try: 
        agent.actor.load_state_dict(actor_best_weight)
        if net_num == 3:  # TD3和SAC有两个评论员网络
            agent.critic_1.load_state_dict(critic_1_best_weight)
            agent.critic_2.load_state_dict(critic_2_best_weight)
        else:
            agent.critic.load_state_dict(critic_best_weight)
    except:
        raise 'please check file route of checkpoints...'    
    end_time = time.time()
    print('总耗时: %d分钟' % ((end_time - start_time)/60))
    # 如果检查点保存了回报列表, 可以不返回
    return return_list


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
    advantage_list = (advantage_list - advantage_list.mean()) / (advantage_list.std() + 1e-5)
    return advantage_list


def show_gym_policy(env_name, model, render_mode='human', epochs=10, steps=500, model_type='AC', if_return=False):
    '''
    `env_name`: 环境名称;\\
    `model`: 类或网络模型, 如agent或agent.net;\\
    `render_mode`: 渲染模式;\\
    `epochs`: 展示轮数\\
    `steps`: 每轮多少步\\
    `model_type`: `V`, `AC`或`AC_n`, 即`价值策略`, `演员评论员`或`强制要求无噪音`;\\    
    `if_return`: 是否返回表, 默认False
    '''
    assert model_type in ['V', 'AC'], '模型类别错误, 应输入 V 或 AC'
    if epochs > 10:
        render_mode == 'rgb_array'
    env = gym.make(env_name, render_mode=render_mode)
    env.reset()
    test_list = []
    with tqdm(total=epochs, leave=False) as pbar:
        for i in range(epochs):  # 测试轮数
            episode_returns = 0
            state = env.reset()[0]  # 第二个输出为info，可以不要
            for _ in range(steps):  # 每回合最多300步
                try:
                    if model_type == 'V':
                        Q_values = model(torch.tensor(state).to('cuda'))
                        action = np.argmax(Q_values.tolist())
                    elif model_type == 'AC':  # 演员评论员框架的梯度策略
                        model.training = False
                        action = model.take_action(state)
                    state, reward, done, truncated, info = env.step(action)
                    episode_returns += reward
                except:
                    env.close()
                    raise Exception('Action execution error!')
                if done or truncated:
                    break
            test_list.append(episode_returns)

            if (i + 1) % 5 == 0:
                pbar.set_postfix(
                    {'epoch': '%d' % (i), 'recent_return': '%.3f' % np.mean(test_list[-5:])})
            pbar.update(1)
    env.close()
    print('平均回报: ', np.mean(test_list).round(3))
    model.training = True  # 演示完毕还原训练状态
    pic_name = model.__class__.__name__  # 获得类名
    if if_return:
        move_avg = 0 if epochs <= 10 else 10
        picture_return(test_list, pic_name, env_name, move_avg)
    return test_list
