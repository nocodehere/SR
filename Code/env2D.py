import numpy as np
import random

class env2D():
    """
    Main class for 2D environments
    """
    def __init__(self, task_name, seed = 1):
        self.seed = random.seed(seed)
        self.task_name = task_name
        self.reward = 0
        self.done = 0
        self.useWall = 0
        self.reset()

    def reset(self):
        """
        Create/reset environment
        :return: 
        """
        if self.task_name == "task3x3":
            self.task3x3()
        elif self.task_name == "task4x4":
            self.task4x4()
        elif self.task_name == "task6x6":
            self.task6x6()
        elif self.task_name == "task7x7_DSRLpaper":
            self.task7x7_DSRLpaper()
        else:
            print("Task does not exist")

    def observation(self):
        """
        returns current state
        :return: self.state
        """
        to_return = self.grid.copy()
        to_return[self.agent[0],self.agent[1]] = 1
        return to_return

    def observation_flat(self):
        """
        returns current state
        :return: self.state flattened
        """
        to_return = self.grid.copy()
        to_return[self.agent[0],self.agent[1]] = 1
        return to_return.flatten()

    def observation_pic_gen(self):
        """
        Generates background picture of not changing objects
        :return: 
        """
        pass

    def observation_pic(self):
        """
        Generates picture of current state
        :return: 
        """
        pass

    def task3x3(self):
        """
        Initialize task0 with preset parameters
        :return: 
        """
        self.grid = np.zeros([3,3])
        #starting location of an agent
        self.agent = np.array([0,0])
        #left, right, up, down
        self.action_space = np.array([0,1,2,3])
        self.actual_move = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]])

    def task4x4(self):
        """
        Initialize task0 with preset parameters
        :return: 
        """
        self.grid = np.zeros([4,4])
        #starting location of an agent
        self.agent = np.array([0,0])
        #left, right, up, down
        self.action_space = np.array([0,1,2,3])
        self.actual_move = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]])

    def task6x6(self):
        """
        Initialize task0 with preset parameters
        :return: 
        """
        self.grid = np.zeros([6,6])
        #starting location of an agent
        self.agent = np.array([0,0])
        #left, right, up, down
        self.action_space = np.array([0,1,2,3])
        self.actual_move = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]])

    def task7x7_DSRLpaper(self):
        self.grid = np.zeros([7,7])
        #left, right, up, down
        self.action_space = np.array([0,1,2,3])
        self.actual_move = np.array([[0,-1],[0,1],[-1,0],[1,0]])
        self.useWall = 1
        #list of walls
        self.walls = np.array([[0,3], [2,3], [3,3], [3,4], [3,5], [3,6],
                               [4,1], [4,2], [4,3], [5,3]])
        #RANDOM starting location of an agent
        self.agent = np.array([0,0])

    def step(self, action):
        new_loc = self.actual_move[action] + self.agent
        if (0 <= new_loc[0] < self.grid.shape[0]) and (0 <= new_loc[1] < self.grid.shape[1]):
            if self.useWall == 1:
                if np.max(np.sum(new_loc == self.walls, axis=1))!=2:
                    self.agent = new_loc
            else:
                self.agent = new_loc
        return self.observation_flat(), self.reward, self.done