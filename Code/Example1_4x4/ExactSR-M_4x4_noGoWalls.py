"""
M for 4x4 matrix (skip env steps returning same position
"""
import numpy as np
import random
from env2D import env2D
from replay_buffer import ReplayBuffer


def one_hot(num, maximum):
    one_hot = np.zeros(maximum)
    one_hot[num] = 1
    return one_hot

M = np.zeros((16, 16))  # for starting and end states (s,s')
GAMMA = 0.95
ALPHA = 0.05

#generate buffer of 100000 random moves in the environment
RANDOM_SEED = 1
ENVIRONMENT = 'task4x4'
BUFFER_SIZE = 100000
MINIBATCH_SIZE = 1
MAX_EPISODE_LENGTH = 150000
TRAIN_STEPS = 1000000

env = env2D(ENVIRONMENT)
np.random.seed(RANDOM_SEED)
env.seed = random.seed(RANDOM_SEED)
replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

env.reset()
for episode_step in xrange(MAX_EPISODE_LENGTH):
    s1 = env.observation_flat()
    action = np.random.choice(4)
    s2, reward, done = env.step(action)
    #add only if not Wall
    if np.logical_not(np.array_equal(s1,s2)):
        replay_buffer.add(s1, 1, 1, 1, s2)
#replay_buffer.size()

#sample from the transitions list and update M
for i in range (TRAIN_STEPS+1):
    if (i % (TRAIN_STEPS/100)) == 0:
        print ('Training step: ' + str(i) + ' of ' + str(TRAIN_STEPS))
    #random move
    s1, a_batch, r_batch, t_batch, s2 = \
        replay_buffer.sample_batch(MINIBATCH_SIZE)

    argmaxs1 = np.argmax(s1)
    TDM = s1 + GAMMA*M[np.argmax(s2),:] - M[argmaxs1,:]
    M[argmaxs1,:] = M[argmaxs1] + ALPHA*TDM

np.set_printoptions(precision=2)
# np.reshape(M[0],(4,4))
# np.reshape(M[3],(4,4))
# np.reshape(M[12],(4,4))
# np.reshape(M[15],(4,4))

np.reshape(M[4],(4,4))
#right
np.array([ [ 1.31,  2.57,  1.55,  0.8 ],
           [ 1.5 ,  2.07,  1.64,  1.05],
           [ 1.15,  1.46,  1.24,  0.83],
           [ 0.64,  0.89,  0.82,  0.5 ]])
#down
np.array([[ 1.27,  1.35,  1.05,  0.61],
       [ 2.65,  2.05,  1.43,  0.89],
       [ 1.68,  1.66,  1.23,  0.78],
       [ 0.91,  1.07,  0.87,  0.5 ]])
