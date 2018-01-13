"""
7x7 with walls as in SRDL paper. NO w.
"""
import numpy as np
import random
from env2D import env2D
from replay_buffer import ReplayBuffer


def one_hot(num, maximum):
    one_hot = np.zeros(maximum)
    one_hot[num] = 1
    return one_hot

M = np.zeros((49, 49))  # for starting and end states (s,s')
GAMMA = 0.95
ALPHA = 0.1

#generate buffer of 100000 random moves in the environment
RANDOM_SEED = 1
ENVIRONMENT = 'task7x7_DSRLpaper'
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
    TDM = s1[0] + GAMMA*M[np.argmax(s2),:] - M[argmaxs1,:]
    M[argmaxs1,:] = M[argmaxs1,:] + ALPHA*TDM

np.set_printoptions(precision=2)
print(np.round(100*np.reshape(M[0,:],(7,7)))/100,2)

#jeigu simetrinis - no causality, kaip 2D atveju, tai tiesiog M vidurkis pasako kelia automatiskai???
print(np.round(100*np.reshape((M[2,:])+(2*M[44,:]),(7,7)))/100,2)

np.sum(np.round(100*np.reshape((M[10,:])*(2*M[44,:]),(7,7)))/100)
np.sum(np.round(100*np.reshape((M[7,:])*(2*M[44,:]),(7,7)))/100)

#
# a = M[40,:]
# indS0 = [4,5,6,11,12,13,18,19,20]
# indS1 = [0,1,2,7,8,9,14,15,16,21,22,23]
# indS2 = [35,36,37,42,43,44]
# indS3 = [32,33,34,39,40,41,46,47,48]
#
# s0 = np.mean([a[indS0]])
# s1 = np.mean([a[indS1]])
# s2 = np.mean([a[indS2]])
# s3 = np.mean([a[indS3]])
