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
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done, truncated):
        self.buffer.append(
            (state, action, reward, next_state, done, truncated))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, truncated = zip(*transitions)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), done, truncated

    def size(self):
        return len(self.buffer)


def picture_return(return_list, policy_name, env_name, move_avg=10):
    sns.set()
    return_list = pd.Series(return_list)
    alpha = 0.5 if move_avg else 1
    return_list.plot(alpha=alpha, label='origin_data')
    if move_avg:
        return_list.rolling(window=move_avg).mean().plot(label='mv_avg_of_%d' % move_avg, linewidth=1.5)
    plt.title('%s on %s' % (policy_name, env_name), fontsize=13)
    plt.xlabel('Epochs', fontsize=13)
    plt.ylabel('Total_return', fontsize=13)
    plt.legend(fontsize=13)
    plt.savefig('pictures/%s on %s.jpg' % (policy_name, env_name))


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] -
              cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def train_on_policy_agent(env, agent, s_epoch, total_epochs, s_episode, total_episodes, return_list, ckp_path):
    start_time = time.time()
    best_score = -1e10  # init_score
    if not return_list:
        return_list = []
    for epoch in range(s_epoch, total_epochs):
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
            s_episode = 0
    try:        
        agent.actor.load_state_dict(actor_best_weight)
        agent.critic.load_state_dict(critic_best_weight)
    except:
        raise 'please check file route of checkpoints...'
    end_time = time.time()
    print('Total: %i mins' % ((end_time - start_time) / 60))
    return return_list


def train_off_policy_agent(env, agent, s_epoch, total_epochs, s_episode, total_episodes, replay_buffer, 
                           minimal_size, batch_size, return_list, ckp_path, net_num=2):

    assert net_num >= 2, 'AC need at least 2 nets...'
    assert (s_epoch <= total_epochs) or (s_episode <= total_episodes), 'end is smaller then begin...'

    start_time = time.time()
    if not return_list:
        return_list = []
    best_score = -1e10
    for epoch in range(s_epoch, total_epochs):
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
                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d, b_t = replay_buffer.sample(
                        batch_size)
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
                    
            s_episode = 0
    try:        
        agent.actor.load_state_dict(actor_best_weight)
    except:
        raise 'please check file route of checkpoints...'
    if net_num == 3:
        agent.critic_1.load_state_dict(critic_1_best_weight)
        agent.critic_2.load_state_dict(critic_2_best_weight)
    else:
        agent.critic.load_state_dict(critic_best_weight)
        
    end_time = time.time()
    print('Total: %d mins' % ((end_time - start_time) / 60))
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    advantage_list = torch.tensor(np.array(advantage_list), dtype=torch.float)
    advantage_list = (advantage_list - advantage_list.mean()) / (advantage_list.std() + 1e-5)
    return advantage_list


def show_gym_policy(env_name, model, render_mode='human', epochs=10, steps=500, model_type='AC', if_return=False):
    assert model_type in ['V', 'AC'], 'model type error, please input V or AC'
    if epochs > 10:
        render_mode == 'rgb_array'
    env = gym.make(env_name, render_mode=render_mode)
    env.reset()
    test_list = []
    for i in range(epochs):
        episode_returns = 0
        state = env.reset()[0]
        for _ in range(steps):
            try:
                if model_type == 'V':
                    Q_values = model(torch.tensor(state).to('cuda'))
                    action = np.argmax(Q_values.tolist())
                elif model_type == 'AC':
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

    env.close()
    print('avg_return: ', np.mean(test_list).round(3))
    model.training = True
    pic_name = model.__class__.__name__
    if if_return:
        move_avg = 0 if epochs <= 10 else 10
        picture_return(test_list, pic_name, env_name, move_avg)
    return test_list
