"""
Be topdown sr:
    Last batch train loss (using only train_net): 0.024586
    mse from real: 0.110244577146

Hardcoded SR1 
Be hidden layer SR_topdown inpute:
    Last batch train loss (using only train_net): 0.0331184
    mse from real: 0.082792694664

Su:
hidden_size_sr_topdown = 10:
    Last batch train loss (using only train_net): 0.0417558
    mse from real: 0.123714916629
hidden_size_sr_topdown = 3:
    Last batch train loss (using only train_net): 0.0301645
    mse from real: 0.0924053160878

"""

import random
import tensorflow as tf
import numpy as np
import numpy.matlib
from env2D import env2D
from replay_buffer import ReplayBuffer

# Training steps
TOTAL_EPISODES = 45
# One episode length
MAX_EPISODE_LENGTH = 200
# learning rate
LEARNING_RATE = 0.0008 #8
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

RANDOM_SEED = 11
# Size of replay buffer
BUFFER_SIZE = 10000
MINIBATCH_SIZE = 64
ENVIRONMENT = 'task7x7_DSRLpaper'

def selu(z,
         scale=1.0507009873554804934193349852946,
         alpha=1.6732632423543772848170429916717):
    return scale * tf.where(z >= 0.0, z, alpha * tf.nn.elu(z))

# got from exact calculation of M
SR_L1_raw = np.array([ [1.77, 0.25, 0.035, 0.006],
                       [0.25, 1.3, 0.15, 0.03],
                       [0.03, 0.26, 1.75, 0.49],
                       [0.004, 0.03, 0.25, 1.94]]) #areas 0,1,2,3

#normalized
SR_L1_norm = (SR_L1_raw - np.mean(SR_L1_raw))/np.std(SR_L1_raw)

class Network(object):
    """
    """
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.SR_L1_dim = 4
        self.learning_rate = learning_rate
        self.tau = tau
        self.w = np.ones(self.state_dim) #good for now
        #self.w = one_hot(5, self.state_dim) #tetsing in 6x6 environment

        # Temp network
        self.inputs, self.action, self.sr_L1, self.out = self.create_network(scope = 'temp_net')
        self.network_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='temp_net')

        # Target Network, for keeping updated
        self.target_inputs, self.target_action, self.target_sr_L1, self.target_out = self.create_network(scope = 'target_net')
        self.target_network_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau)
                                                  + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # curr_state
        self.curr_state = tf.placeholder(tf.float32, [None, self.state_dim])

        self.loss = tf.reduce_mean((tf.square(self.curr_state + GAMMA*self.target_out - self.out)))

        #just for testing
        self.a = tf.placeholder(tf.float32, [None, self.state_dim])
        self.b = tf.placeholder(tf.float32, [None, self.state_dim])
        self.test_loss_var = tf.reduce_mean(tf.square(self.a-self.b))

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

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
        SR_L1 = tf.placeholder(tf.float32, shape=(None, self.SR_L1_dim))

        hidden_size_s = 3 #3
        hidden_size_sr = 3 #3
        hidden_size = 20  # 20
        hidden_size_sr_topdown = 3  # 20

        with tf.variable_scope(scope):
            W_s = tf.Variable(tf.truncated_normal([self.state_dim, hidden_size_s]))
            Wh_s = tf.Variable(tf.truncated_normal([hidden_size_s, hidden_size_sr]))
            W_a = tf.Variable(tf.truncated_normal([self.action_dim, hidden_size_sr]))
            Wh_SR_L1 = tf.Variable(tf.truncated_normal([self.SR_L1_dim, hidden_size_sr_topdown]))
            W_SR_L1 = tf.Variable(tf.truncated_normal([hidden_size_sr_topdown, hidden_size_sr]))
            b_hsr_L1 = tf.Variable(tf.zeros([hidden_size_sr_topdown]))
            b_sr = tf.Variable(tf.zeros([hidden_size]))
            b_hsr = tf.Variable(tf.zeros([hidden_size_sr]))
            b_s = tf.Variable(tf.zeros([hidden_size_s]))
            Wh_sr = tf.Variable(tf.truncated_normal([hidden_size_sr, hidden_size]))
            Wsr_out = tf.Variable(tf.truncated_normal([hidden_size, self.state_dim]))

        Ls = selu(tf.matmul(inputs, W_s) + b_s)
        Lsr_L1 = selu(tf.matmul(SR_L1, Wh_SR_L1) + b_hsr_L1)
        sr_in = selu(tf.matmul(Ls, Wh_s) + tf.matmul(action, W_a)+ tf.matmul(Lsr_L1, W_SR_L1) + b_hsr)
        sr_out = selu(tf.matmul(sr_in, Wh_sr) + b_sr)
        out = tf.matmul(sr_out, Wsr_out)

        return inputs, action, SR_L1, out

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def execute(self, inputs, action, sr_L1):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action,
            self.target_sr_L1: sr_L1
        })

    def train(self, s1, a1, s2, a2, sr_L1, target_sr_L1):
        a, acc, loss  = self.sess.run([self.train_temp_net, self.merged, self.loss], feed_dict={
            self.inputs: s1,
            self.action: a1,
            self.target_inputs: s2,
            self.target_action: a2,
            self.curr_state: s1,
            self.sr_L1: sr_L1,
            self.target_sr_L1:target_sr_L1
        })
        #print loss
        return acc, loss

    def argmax_a(self, inputs, sr2_L1_batch):
        #  batch_size*net.action_dim
        inputs_x_actions = np.matlib.repmat(inputs, self.action_dim, 1)
        sr_L1_x_actions = np.matlib.repmat(sr2_L1_batch, self.action_dim, 1)

        # create action batch
        all_actions = np.zeros((MINIBATCH_SIZE*self.action_dim, self.action_dim))
        for i in range(self.action_dim):
            all_actions[MINIBATCH_SIZE*i:MINIBATCH_SIZE*(i+1),i]=1

        # get SR
        SR = self.sess.run(self.out, feed_dict={
            self.inputs: inputs_x_actions,
            self.action: all_actions,
            self.sr_L1: sr_L1_x_actions
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

def one_hot(num, maximum):
    one_hot = np.zeros(maximum)
    one_hot[num] = 1
    return one_hot

def form_SR_l1_batch(s_batch):
    #one of four regions
    indS0 = [4,5,6,11,12,13,18,19,20]
    indS1 = [0,1,2,7,8,9,14,15,16,21,22,23]
    indS2 = [35,36,37,42,43,44]
    indS3 = [32,33,34,39,40,41,46,47,48]
    s_batch_result = np.zeros((s_batch.shape[0], 4))
    for i in range(s_batch.shape[0]):
        pos = np.argmax(s_batch, 1)[i]
        if pos in indS0:
            s_batch_result[i,:] = SR_L1_norm[0]
        elif pos in indS1:
            s_batch_result[i,:] = SR_L1_norm[1]
        elif pos in indS2:
            s_batch_result[i,:] = SR_L1_norm[2]
        elif pos in indS3:
            s_batch_result[i,:] = SR_L1_norm[3]
        #bottleneck states
        elif pos == 10:
            s_batch_result[i,:] = (SR_L1_norm[0] + SR_L1_norm[1]) / 2
        elif pos == 28:
            s_batch_result[i,:] = (SR_L1_norm[1] + SR_L1_norm[2]) / 2
        elif pos == 45:
            s_batch_result[i,:] = (SR_L1_norm[2] + SR_L1_norm[3]) / 2
    return s_batch_result


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

                sr1_L1_batch = form_SR_l1_batch(s_batch)
                sr2_L1_batch = form_SR_l1_batch(s2_batch)

                # a' = argmax(tempNet(s2)*w)
                a2_batch = net.argmax_a(s2_batch, sr2_L1_batch)
                # use a to update temp net from batch
                acc, loss = net.train(s_batch, a_batch, s2_batch, a2_batch, sr1_L1_batch, sr2_L1_batch)
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
    test_sr = form_SR_l1_batch(np.reshape(test_state, (1, len(test_state))))
    test_sr2 = form_SR_l1_batch(np.reshape(test_state2, (1, len(test_state2))))
    test_sr3 = form_SR_l1_batch(np.reshape(test_state3, (1, len(test_state3))))

    net.sess = sess

    # patikrinti ar Scurrent + y*SR(S2) == SR(S1), nes dabar zuriu tik SR(S1)
    SR = net.execute(np.reshape(test_state, (1, net.state_dim)),
                np.reshape(test_action, (1, net.action_dim)), test_sr)

    SR2 = net.execute(np.reshape(test_state2, (1, net.state_dim)),
                    np.reshape(test_action2, (1, net.action_dim)), test_sr2)

    SR3 = net.execute(np.reshape(test_state3, (1, net.state_dim)),
                    np.reshape(test_action3, (1, net.action_dim)), test_sr3)

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

    sr1_L1_batch = form_SR_l1_batch(s_batch)
    sr2_L1_batch = form_SR_l1_batch(s2_batch)

    # a' = argmax(tempNet(s2)*w)
    a2_batch = net.argmax_a(s2_batch, sr2_L1_batch)

    for i in range(1000):
        net.update_target_network()

    acc, loss = net.train(s_batch, a_batch, s2_batch, a2_batch, sr1_L1_batch, sr2_L1_batch)
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