import gym
import numpy as np
import tensorflow as tf
from q_net import q_net
import matplotlib.pyplot as plt

num_episodes = int(1e3)

env = gym.make('MsPacman-v0')
env.seed(0)

learning_network = q_net('learning')
target_network = q_net('target')
saver = tf.train.Saver()

mspacman_color = 210 + 164 + 74
def preprocess(obs):
    img = obs[1:176:2, ::2] # crop and downsize
    img = img.sum(axis=2) # to greyscale
    img[img==mspacman_color] = 0 # Improve contrast
    img = (img // 3 - 128).astype(np.int8) # normalize from -128 to 127
    return img.reshape(88, 80)

def create_new_state(obs):
    new_state = np.stack([obs, obs, obs, obs], axis=2)
    return new_state

def roll_state(state, obs):
    new_state = np.roll(state, 1, axis=2)
    new_state[:, :, 0] = obs
    return new_state

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, 'partially_trained_model/model.ckpt')

    obs = env.reset()
    obs = preprocess(obs)
    state = create_new_state(obs)

    rewards_tracker = []

    for iteration in range(num_episodes):
        rewards = []

        while True:
            state_input = np.stack([state]).astype(dtype=np.float32)  # prepare to feed placeholder learning_network.obs
            Q = learning_network.get_Q(obs=state_input)
            act = np.argmax(Q)

            next_obs, reward, done, _ = env.step(act)

            rewards.append(reward)

            if done:
                obs = env.reset()
                obs = preprocess(obs)
                state = create_new_state(obs)

                break
            else:
                next_obs = preprocess(next_obs)
                new_state = roll_state(state, next_obs)
                state = new_state

        rewards_tracker.append(sum(rewards))
        smoothed_rewards_tracker = [np.mean(rewards_tracker[max(0, i - 10):i + 1]) for i in range(len(rewards_tracker))]

        plt.figure(figsize=(8, 6))
        plt.plot(rewards_tracker, 'b.', label='total reward')
        plt.plot(smoothed_rewards_tracker, 'r', label='smoothed')
        plt.xlabel('episode number', fontsize=20)
        plt.legend(loc='upper right', prop={'size': 20})
        plt.savefig('test_rewards.png')
        plt.close()

        print(iteration, sum(rewards))
