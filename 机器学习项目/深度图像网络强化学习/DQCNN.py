
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from environment import stock
import time

tf.config.list_logical_devices()


def show_total_reward(rewards):    
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.figure(figsize=(8, 4))
    plt.plot(rewards)
    plt.xlabel("episode", fontsize=14)
    plt.ylabel("total_reward", fontsize=14)
    plt.grid(True)



tf.random.set_seed(42)  # extra code ¨C ensures reproducibility on the CPU

n_outputs = 5  # == env.action_space.n

model = keras.models.Sequential(
    [
        keras.layers.Conv2D(
            16, 6, activation="relu", padding="same", input_shape=[575, 800, 3]
        ),
        keras.layers.MaxPool2D(2), 
        keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(n_outputs),
    ]
)



def epsilon_greedy_policy(model, state, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs) 
    else:
        Q_values = model.predict(state[np.newaxis], verbose=0)[
            0
        ] 
        return Q_values.argmax()



from collections import deque

replay_buffer = deque(maxlen=4000) 


def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_buffer), size=batch_size)
    batch = [replay_buffer[index] for index in indices] 
    return [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)
    ] 

def play_one_step(model, env, state, epsilon):
    action = epsilon_greedy_policy(model, state, epsilon)
    next_state, reward, done = env.step(action)
    replay_buffer.append((state, action, reward, next_state, done))
    return next_state, reward, done

np.random.seed(42)
tf.random.set_seed(42)
rewards = []
best_score = 0
batch_size = 32
discount_factor = 0.95
optimizer = tf.keras.optimizers.Nadam(learning_rate=1e-2)  
loss_fn = tf.keras.losses.mean_squared_error  

env = stock()



target = tf.keras.models.clone_model(model) 
target.set_weights(model.get_weights())


def training_step(batch_size):
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences

    next_Q_values = model.predict(next_states, verbose=0)  # ¡Ù target.predict()
    best_next_actions = next_Q_values.argmax(axis=1) 
    next_mask = tf.one_hot(best_next_actions, n_outputs).numpy()
    max_next_Q_values = (target.predict(next_states, verbose=0) * next_mask).sum(axis=1)

    runs = 1.0 - dones
    target_Q_values = rewards + runs * discount_factor * max_next_Q_values
    target_Q_values = target_Q_values.reshape(-1, 1)
    mask = tf.one_hot(actions, n_outputs)
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


import gc
import scalene


epi = 20
for episode in range(epi):
    if episode >= 49 and (episode + 1) % 10 == 0:
        start_time = time.time()
    obs = env.reset(k_num=50)
    total_reward = 0
    for step in range(50):
        epsilon = max(1 - episode / (epi / 2), 0.01)
        obs, reward, done = play_one_step(model, env, obs, epsilon)
        total_reward += reward
        if done:
            break

    rewards.append(total_reward) 
    if (episode + 1) >= 30:
        recent_reward = np.sum(np.array(rewards[-30:]) > 0) 
        mean_reward = np.mean(rewards) 
    else:
        recent_reward = mean_reward = -1.0
    
    if reward >= best_score:
        best_weights = model.get_weights()
        best_score = reward 
        model.save_weights('DQCNN.h5')

    if episode > 70:
        training_step(batch_size)
        if (episode + 1 % 50) == 0:
            target.set_weights(model.get_weights())
            
    if (episode + 1) % 30 == 0 and (episode + 1) >= 60:
        end_time = time.time()
        period_time = end_time - start_time
        estimated_left_time = float(
            (epi - 1 - episode) * period_time / 1800
        ) 
    elif episode <= 59:
        estimated_left_time = -1.0

    print(f"\r<{episode + 1}>, steps: {step + 1}, ¦Å: {epsilon:.3f}, total_reward: {total_reward:.2f}, avg_reward: {mean_reward:.2f}, [{recent_reward}/30], left: {estimated_left_time:.1f}min          ", end="")
    gc.collect()

model.set_weights(best_weights)

show_total_reward(rewards)



