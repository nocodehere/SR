"""
M for row of six - always going right
"""

import numpy as np

def one_hot(num, maximum):
    one_hot = np.zeros(maximum)
    one_hot[num] = 1
    return one_hot

M = np.zeros((6, 6))  # for starting and end states (s,s')
GAMMA = 0.7
ALPHA = 0.05

# let's say we have buffer of moves [s0,s1], a is always right
movement = np.array(([0,1],[1,2],[2,3],[3,4],[4,5],[5,5]))

#sample from the movement list and update M
for i in range (10000):
    #random move
    moveNo = np.random.choice(len(movement))

    TDM = one_hot(movement[moveNo][0], 6) + GAMMA*M[movement[moveNo][1],:] - M[movement[moveNo][0],:]
    M[movement[moveNo][0],:] = M[movement[moveNo][0]] + ALPHA*TDM

# So M, given s=0 and a=1 (in testing NN based approach) should look like:
M[0,:]