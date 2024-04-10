#fundamental tools for the game environment model
import numpy as np
import math
import random

# for visualations in jupyter
import matplotlib
import matplotlib.pyplot as plt

class exploreGame:
    def __init__(self, size=21):
        #limit size to be between 11 - 101 and odd
        #note that viewer may struggle with large sizes due to grid lines
        if size > 101:
            size = 101
        elif size % 2 == 0:
            size += 1
        if size < 11:
            size = 11
        self.size = int(size)
        self.board = np.zeros((size, size)) # this will be used for obstacles
        self.agent = [int((size -1)/2), int((size -1)/2)] # given size must be odd, this places the player in the center
        self.goal = self.get_random_goal() # puts the goal in a random starting position
        self.won = False # used for primitive win condition
        self.trace = np.zeros((size,size)) # used for tracking the number of turns spent on each tile in the env
        self.update_trace() # adds the occurence of spawning in the center to trace

    def update_trace(self): # increments the tile in the trace that the agent is currently on.
        self.trace[self.agent[0], self.agent[1]] += 1

    def get_trace(self):
        return self.trace.copy() 

    def update(self, action): # the game logic on a turn execution
        old = [self.agent[0], self.agent[1]] # this will be used for collisions later
        if action == 1: # simple switch to encode actions to movement, collides with invisible wall if on boundary
            self.agent[0] = max(0, self.agent[0] - 1) #up
        elif action == 2:
            self.agent[0] = min(self.size-1, self.agent[0] + 1) #down
        elif action == 3:
            self.agent[1] = max(0, self.agent[1] - 1) #left
        elif action == 4:
            self.agent[1] = min(self.size-1, self.agent[1] + 1) # right

        if not self.won: #if we haven't found the goal this game, check if we are at it
            if self.agent[0] == self.goal[0] and self.agent[1] == self.goal[1]: # if we hit the goal, won is true
                self.won = True
                # we will want to add new goals for later
        # if action is 0 = do nothing
        self.update_trace()

    def get_random_goal(self): # randomly spawns the goal in on the map
        new_x = np.random.choice(self.size)
        new_y = np.random.choice(self.size)
        mid = (self.size - 1) // 2

        if new_x == mid and new_y == mid: # if the choice would be the middle
            roll = np.random.choice(2)
            list = np.arange(self.size - 1) # list with one less option
            list[int((self.size - 1) // 2)] = self.size - 1 # replace the midpoint with the outer boundary
            if roll == 1: # 50/50 chance to reassign either coordinate
                new_x = np.random.choice(list)
            else:
                new_y = np.random.choice(list)
        return [new_x, new_y]

    def soft_reset(self): # resets the game state but not the trace
        self.board = self.board * 0
        self.agent = [int((self.size - 1) / 2),
                      int((self.size - 1) / 2)]  # given size must be odd, this places the player in the center
        self.goal = self.get_random_goal()
        self.won = False
        self.update_trace()

    def hard_reset(self): # resets the full object state as if a new instance was made
        self.board = self.board * 0
        self.agent = [int((self.size - 1) / 2),
                      int((self.size - 1) / 2)]  # given size must be odd, this places the player in the center
        self.goal = self.get_random_goal()
        self.won = False
        self.trace = self.trace * 0
        self.update_trace()

    def generate_frame(self): # builds an rgb stack based on game state
        img = np.zeros((self.size, self.size, 3)) # rgb stack
        # order matters here for rendering in the case of an overlap
        # lower means higher priority - ie drawn on top
        if not self.won: # we only draw the goal if its available to hit.
            img[self.goal[0], self.goal[1]] = [0, 255, 0]
        img[self.agent[0], self.agent[1]] = [0, 0, 255]
        return img
    
    def draw(self): # simply draws the current state rbg stack to pyplot output for jupyter
        plt.imshow(self.generate_frame().astype("int"))