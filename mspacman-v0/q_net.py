import tensorflow as tf

class q_net:
    def __init__(self, name):
        """
        inputs-
        name: str
        """

        with tf.variable_scope(name):
            self.obs = tf.placeholder(dtype=tf.float32, shape=[None, 88, 80, 4], name='obs')

            conv1 = tf.layers.conv2d(inputs=self.obs, filters=32, kernel_size=8, strides=4, padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            conv3 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())

            flattened = tf.layers.flatten(conv3)

            fc1 = tf.layers.dense(inputs=flattened, units=512, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.Q = tf.layers.dense(inputs=fc1, units=9, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.Q_squeezed = tf.squeeze(self.Q)

            self.scope = tf.get_variable_scope().name

    def get_Q(self, obs):
        return tf.get_default_session().run(self.Q_squeezed, feed_dict={self.obs: obs})

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
