#fundamental tools for the game environment model
import numpy as np
import math
import random

# for visualations
import matplotlib
import matplotlib.pyplot as plt

class exploreGame:
    def __init__(self, size=21):
        
        if size > 101:
            size = 101
        elif size % 2 == 0:
            size += 1
        if size < 11:
            size = 11
        self.size = int(size)
        self.board = np.zeros((size, size))
        self.agent = [int((size -1)/2), int((size -1)/2)] # given size must be odd, this places the player in the center
        self.goal = self.get_random_goal()
        self.trace = np.zeros((size,size))
        self.update_trace()

    def update_trace(self):
        self.trace[self.agent[0], self.agent[1]] += 1

    def get_trace(self):
        return self.trace.copy() 

    def update(self, action):
        if action == 1: # simple switch to encode actions to movement, collides with invisible wall if on boundary
            self.agent[0] = max(0, self.agent[0] - 1) #up
        elif action == 2:
            self.agent[0] = min(self.size-1, self.agent[0] + 1) #down
        elif action == 3:
            self.agent[1] = max(0, self.agent[1] - 1) #left
        elif action == 4:
            self.agent[1] = min(self.size-1, self.agent[1] + 1) # right
        # if action is 0 = do nothing
        self.update_trace()

    def get_random_goal(self):
        choice = [np.random.choice(self.size), np.random.choice(self.size)]
        return choice

    def soft_reset(self): # resets the game state but not the trace
        self.board = self.board * 0
        self.agent = [int((self.size - 1) / 2),
                      int((self.size - 1) / 2)]  # given size must be odd, this places the player in the center
        self.goal = self.get_random_goal()
        self.update_trace()

    def hard_reset(self): # resets the full object state as if a new instance was made
        self.board = self.board * 0
        self.agent = [int((self.size - 1) / 2),
                      int((self.size - 1) / 2)]  # given size must be odd, this places the player in the center
        self.goal = self.get_random_goal()
        self.trace = self.trace * 0
        self.update_trace()

    def generate_frame(self):
        img = np.zeros((self.size, self.size, 3)) # rgb stack
        img[self.agent[0], self.agent[1]] = [0, 0, 255]
        img[self.goal[0], self.goal[1]] = [0, 255, 0]
        return img
    
    def draw(self):
        plt.imshow(self.generate_frame().astype("int"))