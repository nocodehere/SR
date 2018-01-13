"""
7x7 with walls as in SRDL paper. Using w.
No autoencoder, Sin * w = R

Last batch train loss (using only train_net): 0.0450134
mse from real: 0.0980146418043

Much better than using AdamOpt, but much slower
"""

import random
import tensorflow as tf
import numpy as np
import numpy.matlib
from env2D import env2D
from replay_buffer import ReplayBuffer

# Training steps
TOTAL_EPISODES = 2500
# One episode length
MAX_EPISODE_LENGTH = 200
# learning rate
LEARNING_RATE = 0.0001 #8
# y discount
GAMMA = 0.95
# target network update param
TAU = 0.01 #0.01
# greedy exploration parameter degrade rate
EPS_DEGRADE = 0.03
# every some steps copy test_net to train_net
COPY_NET_STEPS = 1 #1

# Directory for storing tb summary
SUMMARY_DIR = './results/tf_SR'

RANDOM_SEED = 10
# Size of replay buffer
BUFFER_SIZE = 10000
MINIBATCH_SIZE = 64
ENVIRONMENT = 'task7x7_DSRLpaper'

def selu(z,
         scale=1.0507009873554804934193349852946,
         alpha=1.6732632423543772848170429916717):
    return scale * tf.where(z >= 0.0, z, alpha * tf.nn.elu(z))

class Network(object):
    """
    """
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.w = np.ones(self.state_dim) #good for now
        #self.w = one_hot(5, self.state_dim) #tetsing in 6x6 environment

        # Temp network
        self.inputs, self.action, self.out = self.create_network(scope = 'temp_net')
        self.network_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='temp_net')

        # Target Network, for keeping updated
        self.target_inputs, self.target_action, self.target_out = self.create_network(scope = 'target_net')
        self.target_network_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')

        with tf.variable_scope("w"):
            W_s = tf.Variable(tf.truncated_normal([self.state_dim, self.state_dim]))

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau)
                                                  + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # curr_state
        self.curr_state = tf.placeholder(tf.float32, [None, self.state_dim])

        self.loss = tf.reduce_mean((tf.square(self.curr_state + GAMMA*self.target_out - self.out)) + 0.1*tf.nn.relu(-self.out))

        #just for testing
        self.a = tf.placeholder(tf.float32, [None, self.state_dim])
        self.b = tf.placeholder(tf.float32, [None, self.state_dim])
        self.test_loss_var = tf.reduce_mean(tf.square(self.a-self.b))

        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        #self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

        #for logging
        with tf.name_scope('accuracy'):
            accuracy = self.loss
        tf.summary.scalar('accuracy', accuracy)
        # Merge all the summaries and write them out
        self.merged = tf.summary.merge_all()

        #train only temp network
        train_temp_net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "temp_net")
        self.train_temp_net = self.optimizer.minimize(self.loss, var_list=train_temp_net_vars)

    def create_network(self, scope):
        inputs = tf.placeholder(tf.float32, shape=(None, self.state_dim))
        action = tf.placeholder(tf.float32, shape=(None, self.action_dim))
        hidden_size_sr = 3 #3
        hidden_size = 20 #20

        with tf.variable_scope(scope):
            W_s = tf.Variable(tf.truncated_normal([self.state_dim, hidden_size_sr]))
            W_a = tf.Variable(tf.truncated_normal([self.action_dim, hidden_size_sr]))
            b_sr = tf.Variable(tf.zeros([hidden_size]))
            b_hsr = tf.Variable(tf.zeros([hidden_size_sr]))

            Wh_sr = tf.Variable(tf.truncated_normal([hidden_size_sr, hidden_size]))
            Wsr_out = tf.Variable(tf.truncated_normal([hidden_size, self.state_dim]))


        sr_in = selu(tf.matmul(inputs, W_s) + tf.matmul(action, W_a) + b_hsr)
        sr_out = selu(tf.matmul(sr_in, Wh_sr) + b_sr)
        out = tf.matmul(sr_out, Wsr_out)

        return inputs, action, out

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def execute(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def train(self, s1, a1, s2, a2):
        a, acc, loss  = self.sess.run([self.train_temp_net, self.merged, self.loss], feed_dict={
            self.inputs: s1,
            self.action: a1,
            self.target_inputs: s2,
            self.target_action: a2,
            self.curr_state: s1
        })
        #print loss
        return acc, loss

    def test_loss(self, a, b):
        loss  = self.sess.run([self.test_loss_var], feed_dict={
            self.a: a,
            self.b: b
        })
        return loss

    def argmax_a(self, inputs):
        #  batch_size*net.action_dim
        inputs_x_actions = np.matlib.repmat(inputs, self.action_dim, 1)

        # create action batch
        all_actions = np.zeros((MINIBATCH_SIZE*self.action_dim, self.action_dim))
        for i in range(self.action_dim):
            all_actions[MINIBATCH_SIZE*i:MINIBATCH_SIZE*(i+1),i]=1

        # get SR
        SR = self.sess.run(self.out, feed_dict={
            self.inputs: inputs_x_actions,
            self.action: all_actions
        })

        #calculate Q_val of every candidate
        Q_val = np.matmul(SR, self.w)

        # choose where max Q_val
        max_a = np.zeros((MINIBATCH_SIZE, self.action_dim))
        for i in range(MINIBATCH_SIZE):
            #max_a[i,:] = one_hot(np.argmax(Q_val[i::MINIBATCH_SIZE]), self.action_dim)
            max_a[i,:] = one_hot(np.random.choice(self.action_dim), self.action_dim)
            #max_a[i,:] = one_hot(1, self.action_dim) #always right move

        #create argmaxa batch
        return max_a

# ===========================
#   Tensorflow Summary Ops
# ===========================


# def build_summaries():
#     episode_reward = tf.Variable(0.)
#     tf.summary.scalar("Reward", episode_reward)
#     episode_ave_max_q = tf.Variable(0.)
#     tf.summary.scalar("Qmax Value", episode_ave_max_q)
#
#     summary_vars = [episode_reward, episode_ave_max_q]
#     summary_ops = tf.summary.merge_all()
#
#     return summary_ops, summary_vars


# def train(sess, env, net):
#     # Set up summary Ops
#     # summary_ops, summary_vars = build_summaries()
#
#     sess.run(tf.global_variables_initializer())
#     saver = tf.train.Saver()
#     writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
#
#     # Initialize target networks
#     net.update_target_network()
#
#     env.reset()
#     observation = env.observation()
#     print(observation)
#     action = np.random.choice(env.action_space)
#     print(action)
#     observation, reward, done = env.step(action)

def one_hot(num, maximum):
    one_hot = np.zeros(maximum)
    one_hot[num] = 1
    return one_hot

# def main(_):
with tf.Session() as sess:
    train_step = 0
    env = env2D(ENVIRONMENT)
    np.random.seed(RANDOM_SEED)
    tf.set_random_seed(RANDOM_SEED)
    env.seed = random.seed(RANDOM_SEED)

    state_dim = env.observation_flat().shape[0]
    action_dim = env.action_space.shape[0]

    # create network
    net = Network(sess, state_dim, action_dim, LEARNING_RATE, TAU)

    # train(sess, env, net)

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

    # Initialize target networks
    net.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

    for episode in xrange(TOTAL_EPISODES):
        print ('training episode: ' + str(episode))
        env.reset()
        for episode_step in xrange(MAX_EPISODE_LENGTH):
            s1 = env.observation_flat()
            #env.observation()
            #choose action - if rand < eps: choose action randomly.
            # else: choose argmaxa by network
            if 1==1: #kol nesimoko w, tol tik random vaiksto.
                action = np.random.choice(env.action_space)
                #action = 1
                s2, reward, done = env.step(action)

            else:
                #choose argmax_a
                net.execute(np.reshape(s1, (1, net.state_dim)),
                            np.reshape(np.zeros(net.action_dim), (1, net.action_dim)))

            #if not hit wall
            if np.logical_not(np.array_equal(s1, s2)):
                replay_buffer.add(s1, one_hot(action, net.action_dim), reward, done, s2)


            ## TRAINING
            if replay_buffer.size() > MINIBATCH_SIZE:
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(MINIBATCH_SIZE)


                # a' = argmax(tempNet(s2)*w)
                a2_batch = net.argmax_a(s2_batch)
                # use a to update temp net from batch
                acc, loss = net.train(s_batch, a_batch, s2_batch, a2_batch)
                #print(loss)

                train_step += 1
                writer.add_summary(acc, train_step)
                #update target net
                if (train_step % COPY_NET_STEPS) == 0:
                    net.update_target_network()

    #save session for later testing
    saver.save(sess, "saved_models/SR_task0.ckpt")

## TESTING on target network - load session to test on same results

with tf.Session() as sess:
    #if os.path.isfile("./SR_task0.ckpt"):
    saver.restore(sess, "saved_models/SR_task0.ckpt")
    test_state =   one_hot(0, net.state_dim)
    test_state2 =  one_hot(1, net.state_dim) #state after going right
    test_state3 =  one_hot(0, net.state_dim) #state after going down
    test_action3 = one_hot(3, net.action_dim)  # going down
    test_action =  one_hot(1, net.action_dim)  # going right
    test_action2 = one_hot(1, net.action_dim)  # going right
    net.sess = sess

    # patikrinti ar Scurrent + y*SR(S2) == SR(S1), nes dabar zuriu tik SR(S1)
    SR = net.execute(np.reshape(test_state, (1, net.state_dim)),
                np.reshape(test_action, (1, net.action_dim)))

    SR2 = net.execute(np.reshape(test_state2, (1, net.state_dim)),
                    np.reshape(test_action2, (1, net.action_dim)))

    SR3 = net.execute(np.reshape(test_state3, (1, net.state_dim)),
                    np.reshape(test_action3, (1, net.action_dim)))

    np.set_printoptions(precision=2)
    print('Going right from 0 [state0+y*SR2]: ')
    print(np.round(np.reshape(test_state + GAMMA*SR2, (7, 7)),2)) #going down from 0
    print('Going right from 0 [SR0]: ')
    print(np.round(np.reshape(SR, (7, 7)), 2)) #going right from 0
    print('Going down from 0 [SR3]: ')
    print(np.round(np.reshape(SR3, (7, 7)), 2)) #going down from 0

    #print(loss)
    s_batch, a_batch, r_batch, t_batch, s2_batch = \
        replay_buffer.sample_batch(MINIBATCH_SIZE)

    # a' = argmax(tempNet(s2)*w)
    a2_batch = net.argmax_a(s2_batch)
    # use a to update temp net from batch

    for i in range(1000):
        net.update_target_network()

    acc, loss = net.train(s_batch, a_batch, s2_batch, a2_batch)
    print ('Last batch train loss (using only train_net): ' + str(loss))

    realRight = np.array([ [ 1.14,  2.57,  1.21,  0.  ,  0.18,  0.23,  0.15],
                           [ 1.39,  2.15,  1.64,  0.52,  0.49,  0.32,  0.22],
                           [ 1.04,  1.51,  0.99,  0.  ,  0.19,  0.23,  0.14],
                           [ 0.73,  0.86,  0.58,  0.  ,  0.  ,  0.  ,  0.  ],
                           [ 0.32,  0.  ,  0.  ,  0.  ,  0.02,  0.03,  0.02],
                           [ 0.25,  0.16,  0.08,  0.  ,  0.04,  0.04,  0.02],
                           [ 0.12,  0.15,  0.11,  0.06,  0.05,  0.03,  0.02]])


    realDown = np.array([  [ 1.27,  1.4 ,  0.74,  0.  ,  0.14,  0.19,  0.12],
                           [ 2.62,  2.07,  1.3 ,  0.42,  0.4 ,  0.27,  0.18],
                           [ 1.47,  1.63,  0.93,  0.  ,  0.15,  0.19,  0.12],
                           [ 0.95,  0.96,  0.59,  0.  ,  0.  ,  0.  ,  0.  ],
                           [ 0.4 ,  0.  ,  0.  ,  0.  ,  0.03,  0.03,  0.02],
                           [ 0.32,  0.2 ,  0.1 ,  0.  ,  0.05,  0.05,  0.03],
                           [ 0.15,  0.18,  0.14,  0.07,  0.07,  0.04,  0.02]])

    mse = np.sum(np.square(np.reshape(SR, (7, 7)) - realRight)) / 49
    mse2 = np.sum(np.square(np.reshape(SR3, (7, 7)) - realDown)) / 49

    print('mse from real: ' +str((mse+mse2)/2.))

    ## EIGEN ANALYSIS
    #eigVal, eigVec = numpy.linalg.eig(realDown)
    #eigVec.shape
