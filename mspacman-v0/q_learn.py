import tensorflow as tf

class q_learn:
    def __init__(self, learning_network, target_network, gamma=0.99):
        """
        inputs-
        learning_network:
        target_network:
        gamma: discount factor
        """

        self.learning_network = learning_network
        self.target_network = target_network

        learning_network_params = self.learning_network.get_trainable_variables()
        target_network_params = self.target_network.get_trainable_variables()

        # assign_operations for policy parameter values to old policy parameters
        with tf.variable_scope('assign_op'):
            self.assign_ops = []
            for v_old, v in zip(target_network_params, learning_network_params):
                self.assign_ops.append(tf.assign(v_old, v))

        # inputs for train_op
        with tf.variable_scope('train_inp'):
            self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
            self.rewards = tf.placeholder(dtype=tf.float32, shape=[None], name='rewards')
            self.dones = tf.placeholder(dtype=tf.float32, shape=[None], name='dones')

        # q's of actions which agent took
        learning_Q = self.learning_network.Q
        selected_learning_Q = learning_Q * tf.one_hot(indices=self.actions, depth=learning_Q.shape[1])
        selected_learning_Q = tf.reduce_sum(selected_learning_Q , axis=1)

        target_Q = self.target_network.Q
        max_target_Q = tf.reduce_max(target_Q, axis=1)

        with tf.variable_scope('loss/huber'):
            y = self.rewards + self.dones*gamma*max_target_Q
            loss = tf.losses.huber_loss(selected_learning_Q, y, reduction=tf.losses.Reduction.NONE)
            loss = tf.reduce_mean(loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.train_op = optimizer.minimize(loss, var_list=learning_network_params)

    def train(self, obs, actions, next_obs, rewards, dones):
        tf.get_default_session().run(self.train_op, feed_dict={self.learning_network.obs: obs, self.actions: actions,
                                                                        self.target_network.obs: next_obs,
                                                                        self.rewards: rewards, self.dones: dones})

    def assign_network_parameters(self):
        # assign learning q network parameter values to target network parameters
        return tf.get_default_session().run(self.assign_ops)
