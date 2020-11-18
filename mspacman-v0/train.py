import gym
import numpy as np
import tensorflow as tf
from q_net import q_net
from q_learn import q_learn
import matplotlib.pyplot as plt

mini_batch_size = 32
num_iterations = int(1e7)
gamma = 0.99
max_buffer_size = 1e5
initial_buffer_size = max_buffer_size/5
chop = int(max_buffer_size/100)
update_learn_every_n_iterations = 4
update_target_every_n_iterations = 5000
save_model_every_n_iterations = 1000
validate_every_n_training_episodes = 5
plot_state_every_n_iterations = 100
epsilon_anneal = 1e6
action_repeat = 4

env = gym.make('MsPacman-v0')
env.seed(0)
learning_network = q_net('learning')
target_network = q_net('target')
q_learn = q_learn(learning_network, target_network, gamma=gamma)
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

def epsilon_greedy(Q, epsilon):
    if np.random.uniform() > epsilon:
        act = np.argmax(Q)
    else:
        act = np.random.randint(len(Q))
    return act

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    obs = env.reset()
    obs = preprocess(obs)
    state = create_new_state(obs)

    max_Qs_temp = []
    max_Qs = []

    validation_rewards_temp = []
    validation_rewards = []

    observations_buffer = []
    actions_buffer = []
    next_observations_buffer = []
    rewards_buffer = []
    dones_buffer = []

    # fill buffer
    while True:
        state_input = np.stack([state]).astype(dtype=np.float32)  # prepare to feed placeholder learning_network.obs
        Q = learning_network.get_Q(obs=state_input)

        act = epsilon_greedy(Q, 1.0)

        reward = 0
        for i in range(action_repeat):
            next_obs, r, done, _ = env.step(act)
            reward += r
            if done:
                break

        next_obs = preprocess(next_obs)
        new_state  = roll_state(state, next_obs)

        observations_buffer.append(state)
        actions_buffer.append(act)
        next_observations_buffer.append(new_state)
        rewards_buffer.append(reward)
        if done:
            dones_buffer.append(0)
        else:
            dones_buffer.append(1)

        if len(dones_buffer) == initial_buffer_size:
            break

        if done:
            obs = env.reset()
            obs = preprocess(obs)
            state = create_new_state(obs)
        else:
            state = new_state

    obs = env.reset()
    obs = preprocess(obs)
    state = create_new_state(obs)

    training_episode_counter = 0
    for iteration in range(num_iterations):
        if iteration % plot_state_every_n_iterations == 0:
            plt.subplot(2, 2, 1)
            plt.imshow((state[:, :, 0].astype(dtype=np.float32)+128.0)/255.0)
            plt.subplot(2, 2, 2)
            plt.imshow((state[:, :, 1].astype(dtype=np.float32)+128.0)/255.0)
            plt.subplot(2, 2, 3)
            plt.imshow((state[:, :, 2].astype(dtype=np.float32)+128.0)/255.0)
            plt.subplot(2, 2, 4)
            plt.imshow((state[:, :, 3].astype(dtype=np.float32)+128.0)/255.0)
            plt.savefig('check-state-frames.png')

        state_input = np.stack([state]).astype(dtype=np.float32) # prepare to feed placeholder learning_network.obs
        Q = learning_network.get_Q(obs=state_input)
        max_Qs_temp.append(np.amax(Q))

        act = epsilon_greedy(Q, np.maximum(0.1, 1.0 - iteration/epsilon_anneal))

        reward = 0
        for i in range(action_repeat):
            next_obs, r, done, _ = env.step(act)
            reward += r
            if done:
                break

        next_obs = preprocess(next_obs)
        new_state = roll_state(state, next_obs)

        observations_buffer.append(state)
        actions_buffer.append(act)
        next_observations_buffer.append(new_state)
        rewards_buffer.append(reward)
        if done:
            dones_buffer.append(0)
        else:
            dones_buffer.append(1)

        if len(dones_buffer) == max_buffer_size:
            observations_buffer = observations_buffer[chop:]
            actions_buffer = actions_buffer[chop:]
            next_observations_buffer = next_observations_buffer[chop:]
            rewards_buffer = rewards_buffer[chop:]
            dones_buffer = dones_buffer[chop:]

        if iteration % update_target_every_n_iterations == 0:
            q_learn.assign_network_parameters()

        if iteration % update_learn_every_n_iterations == 0:
            sample_indices = np.random.randint(low=0, high=len(dones_buffer), size=mini_batch_size)

            sample_obs = np.take(observations_buffer, indices=sample_indices, axis=0).astype(dtype=np.float32)
            sample_acts = np.take(actions_buffer, indices=sample_indices, axis=0).astype(dtype=np.int32)
            sample_next_obs = np.take(next_observations_buffer, indices=sample_indices, axis=0).astype(dtype=np.float32)
            sample_rewards = np.take(rewards_buffer, indices=sample_indices, axis=0).astype(dtype=np.float32)
            sample_dones = np.take(dones_buffer, indices=sample_indices, axis=0).astype(dtype=np.float32)

            q_learn.train(obs=sample_obs, actions=sample_acts, next_obs=sample_next_obs, rewards=sample_rewards, dones=sample_dones)

            max_Qs.append(np.amax(max_Qs_temp))
            max_Qs_temp = []

            smoothed_max_Qs = [np.mean(max_Qs[max(0, i - 100):i + 1]) for i in range(len(max_Qs))]

            plt.figure(figsize=(8, 6))
            plt.plot(max_Qs, 'b.', label='max Q')
            plt.plot(smoothed_max_Qs, 'r', label='smoothed')
            plt.xlabel('iteration', fontsize=20)
            plt.legend(loc='upper left', prop={'size': 20})
            plt.savefig('max_Q.png')
            plt.close()

        if iteration % save_model_every_n_iterations == 0:
            saver.save(sess, './partially_trained_model/model.ckpt')
            print('hopefully learning ms pacman better! model saved. iteration: ', iteration)

        if done:
            training_episode_counter += 1

            obs = env.reset()
            obs = preprocess(obs)
            state = create_new_state(obs)

            if training_episode_counter % validate_every_n_training_episodes == 0:
                while True:
                    state_input = np.stack([state]).astype(dtype=np.float32)  # prepare to feed placeholder learning_network.obs
                    Q = learning_network.get_Q(obs=state_input)
                    act = np.argmax(Q)
                    next_obs, reward, done, _ = env.step(act)

                    validation_rewards_temp.append(reward)

                    if done:
                        validation_rewards.append(np.sum(validation_rewards_temp))
                        validation_rewards_temp = []

                        plt.figure(figsize=(8, 6))
                        plt.plot(validation_rewards, 'bs')
                        plt.xlabel('after 5n training episodes', fontsize=20)
                        plt.ylabel('total reward', fontsize=20)
                        plt.savefig('validation_rewards.png')
                        plt.close()

                        break
                    else:
                        next_obs = preprocess(next_obs)
                        new_state = roll_state(state, next_obs)
                        state = new_state

                obs = env.reset()
                obs = preprocess(obs)
                state = create_new_state(obs)
        else:
            state = new_state
