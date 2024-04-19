# fundamental tools for the game environment model
import numpy as np
import os
import pygame
from randomchooser import Chooser
# for visualizations in jupyter
import matplotlib.pyplot as plt


class ExploreGame:
    def __init__(self, size=21, maze=True):
        # limit size to be between 11 - 101 and odd
        # note that viewer may struggle with large sizes due to grid lines
        if size > 101:
            size = 101
        elif size % 2 == 0:
            size += 1
        if size < 11:
            size = 11
        # util structs
        self.action_pairs = np.array([[-1, 0], [0, -1], [1, 0], [0, 1]])  # the action vectors excluding no movement
        self.size = int(size)
        self.maze = maze
        self.board = np.zeros((size, size))  # this will be used for obstacles
        if maze:
            self.build_map()
        self.agent = [int((size - 1) / 2),
                      int((size - 1) / 2)]  # given size must be odd, this places the player in the center
        self.goal_options = None
        self.goal = self.get_random_goal()  # puts the goal in a random starting position
        self.won = False  # used for primitive win condition
        self.trace = np.zeros((size, size))  # used for tracking the number of turns spent on each tile in the env
        self.update_trace()  # adds the occurence of spawning in the center to trace



    def gen_noise_pattern(self):  # generate a noise pattern of spawn points for wall chains
        return np.random.choice([0, 5, 4, 3], (self.size, self.size), p=[.9, .025, .025, .05])

    def action_switch(self, coord, action):
        return coord - self.action_pairs[action]

    def propagate_wall(self, coord, val, last_move):  # takes a root location, and build val number of
        # leaves recursively, does not allow for doubling back with last_move and detection
        if val == 0:
            return True
        else:
            probs = np.ones(4)
            if last_move < 4:
                probs[(last_move + 2) % 4] = 0  # disallow going backwards
            probs = probs/probs.sum()  # normalize
            action = np.random.choice([0, 1, 2, 3], p=probs)
            target = self.action_switch(coord, action)
            # stay in bounds
            target[0] = max(0, min(target[0], 20))
            target[1] = max(0, min(target[1], 20))

            if self.board[target[0]][target[1]] == 0:  # if the space does not have an obstacle yet
                self.board[coord[0]][coord[1]] = 1  # place a wall
                self.propagate_wall(target, val-1, action)  # Recurr
            elif self.board[target[0]][target[1]] == 1:  # if the stem has intersected an existing wall
                self.propagate_wall(target, val, action)  # Skip this position and recurr

    def find_diagonal_choices(self, i, j, window):
        if window.sum() == 2:
            mytrace = window.trace()
            if mytrace == 2:  # A left diagonal
                return [[i, j], [i+1, j+1]]
            elif mytrace == 0:  # a right diagonal
                return [[i, j+1], [i + 1, j]]
            else:
                return None
        else:
            return None

    def correct_diagonal_wall(self, i, j, p=.5):
        action = np.random.choice([0, 1], p=[1-p, p])  # Choose between adding or removing a wall to correct diagonality
        if action == 1:
            self.board[i, j] = 1  # add a wall
        else:
            self.board[i, j] = 0  # remove a wall

    def clean_diagonal_walls(self, p=.5):  # We want to remove instances of obstructions composed of diagonal walls
        # We are going to convolve looking for strictly diagonal obstructions
        complete = True
        for i in range(self.size - 1):  # -1 as we are suing a 2x2 convol
            for j in range(self.size - 1):
                window = self.board[i:i+2, j:j+2]
                choices = self.find_diagonal_choices(i, j, window)

                if choices is not None:  # must be true for a strictly diagonal wall to be present
                    # must be a left diagonal
                    complete = False
                    choice = np.random.choice([0, 1])
                    choice = choices[choice]
                    self.correct_diagonal_wall(choice[0], choice[1], p)
        if not complete:  # recurr with more bias to destruction until clean
            self.clean_diagonal_walls(max(0, p - .1))  # clean again but with less additive correction
        return complete

    def build_map(self):
        pattern = self.gen_noise_pattern()  # we need a noise pattern to build from

        # let's do a naive search over the pattern, more efficient solutions exist in numpy but this is legible
        for i in range(self.size):
            for k in range(self.size):
                if pattern[i][k] > 0:
                    val = pattern[i][k]
                    # a function that takes i,k, val to build a stem
                    self.propagate_wall((i, k), val, 4)
        cleaned = self.clean_diagonal_walls()
        center = int((self.size - 1) / 2)
        self.board[center, center] = 0  # Force spawn point to be open
        return pattern, cleaned

    def update_trace(self):  # increments the tile in the trace that the agent is currently on.
        self.trace[self.agent[0], self.agent[1]] += 1

    def get_trace(self):
        return self.trace.copy()

    def update(self, action):  # the game logic on a turn execution
        old = [self.agent[0], self.agent[1]]  # this will be used for collisions later
        if action == 1:  # simple switch to encode actions to movement, collides with invisible wall if on boundary
            self.agent[0] = max(0, self.agent[0] - 1)  # up
        elif action == 2:
            self.agent[0] = min(self.size - 1, self.agent[0] + 1)  # down
        elif action == 3:
            self.agent[1] = max(0, self.agent[1] - 1)  # left
        elif action == 4:
            self.agent[1] = min(self.size - 1, self.agent[1] + 1)  # right

        if not self.won:  # if we haven't found the goal this game, check if we are at it
            if self.agent[0] == self.goal[0] and self.agent[1] == self.goal[1]:  # if we hit the goal, won is true
                self.won = True
                # we will want to add new goals for later
        # if action is 0 = do nothing
        self.update_trace()

    def build_goal_options(self):
        self.goal_options = []
        free_squares = np.where(self.board == 0)
        center = int((self.size - 1) / 2)
        for (i, j) in zip(free_squares[0], free_squares[1]):
            if i == center and j == center:  # Skip where the agent must spawn
                continue
            else:
                self.goal_options.append([i, j])

    def get_random_goal(self):  # randomly spawns the goal in on the map
        if self.goal_options is None:
            self.build_goal_options()

        choice = np.random.choice(np.arange(len(self.goal_options)))

        return self.goal_options[choice]

    def soft_reset(self):  # resets the game state but not the trace or map
        #self.board = self.board * 0
        self.agent = [int((self.size - 1) / 2),
                      int((self.size - 1) / 2)]  # given size must be odd, this places the player in the center
        self.goal = self.get_random_goal()
        self.won = False
        self.update_trace()

    def hard_reset(self):  # resets the full object state as if a new instance was made
        if self.maze:
            self.board = self.board * 0
            self.build_map()
        self.goal_options = None
        self.agent = [int((self.size - 1) / 2),
                      int((self.size - 1) / 2)]  # given size must be odd, this places the player in the center
        self.goal = self.get_random_goal()
        self.won = False
        self.trace = self.trace * 0
        self.update_trace()

    def generate_frame(self):  # builds an rgb stack based on game state
        img = np.zeros((self.size, self.size, 3))  # rgb stack
        # order matters here for rendering in the case of an overlap
        # lower means higher priority - ie drawn on top
        img[self.board == 1] = [128, 96, 96]
        if not self.won:  # we only draw the goal if its available to hit.
            img[self.goal[0], self.goal[1]] = [0, 255, 0]
        img[self.agent[0], self.agent[1]] = [0, 0, 255]
        return img

    def draw(self):  # simply draws the current state rbg stack to pyplot output for jupyter
        plt.imshow(self.generate_frame().astype("int"))


class ExploreViewer:
    def __init__(self, events, name="Explore",x=5, y=20, size=21, window_height=300, framerate=60):
        # this sets the windows location, this will be used to distribute windows when simulating many games at once
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (x, y)
        pygame.init()
        screen = pygame.display.set_mode((100, 100))
        pygame.init()
        self.name = name
        self.event_queue = events
        self.size = size
        self.framerate = framerate
        if not events.empty():
            input = events.get()
            self.env_map = input[0]
            self.trace_map = input[1]
        else:
            self.env_map = np.zeros((self.size, self.size, 3))
            self.trace_map = np.zeros((self.size, self.size))
        self.screen_width = 10 + window_height*2
        self.screen_height = window_height

        self.grid_width = 4 # keep me even
        self.rect_width = (self.screen_height - (self.size+1)*self.grid_width)/self.size #adjusting the size based on parameters

        #colored background for contrast on grid lines
        self.background = pygame.Rect(0,0,self.screen_width,self.screen_height)
        self.background_color = (32,32, 32)

        # for the boundary between the game board and the trace map render regions
        self.boundary_width = 10
        self.zone_boundary = pygame.Rect(self.screen_width//2 - self.boundary_width//2, 0, self.boundary_width, self.screen_height)
        self.boundary_color = (255,255, 0)

        # create a list of rects in a padded grid on the left half of the window for rendering the game state visualization
        self.env_rects = []
        self.env_colors = self.env_map.reshape(-1,3) # this reshape matches the dimensional representation of the rects
        for i in range(self.size**2):
            self.env_rects.append(pygame.Rect((i%self.size)*(self.rect_width+self.grid_width) + self.grid_width, (i//self.size)*(self.rect_width+self.grid_width) + self.grid_width,self.rect_width,self.rect_width))

        # create a list of rects in a padded grid on the right half of the window for rendering the trace heatmap visualization
        self.heatmap_rects = []
        self.heatmap = self.trace_map.flatten() # again to match the dimensionality of the rect list
        if self.heatmap.max() > 0:
            self.heatmap = self.heatmap/self.heatmap.max()*255  # scaling between 0-255 without normalizing fully. This ensures the same relative scale accross any 2 heatmaps
        for i in range(self.size**2):                       # Note: this only aligns the lower bound at 0, bright spots may have different magnitudes. But this allows us to ensure destinctions
                                                            # We want to emphasize the differences in UNEXPLORED areas as opposed to simply distributions
            self.heatmap_rects.append(pygame.Rect((i%self.size)*(self.rect_width+self.grid_width) + self.grid_width + self.screen_width//2 + self.boundary_width//2
                                             , (i//self.size)*(self.rect_width+self.grid_width) + self.grid_width,self.rect_width,self.rect_width))
        # Set up display
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption(self.name)
        self.run = True

        # draw first game state
        pygame.draw.rect(self.screen, self.background_color, self.background)
        pygame.draw.rect(self.screen, self.boundary_color, self.zone_boundary)
        for (rect, color) in zip(self.env_rects, self.env_colors):
            pygame.draw.rect(self.screen, (color[0], color[1], color[2]), rect)

        for (rect, intensity) in zip(self.heatmap_rects, self.heatmap):
            pygame.draw.rect(self.screen, (intensity, intensity, intensity), rect)

        # update the actual screen
        pygame.display.update()
        pygame.time.wait((1000 // self.framerate))


    def start(self): # enters the animating loop, waiting for new frames to be passed into queue and render the info
        while self.run:
            if not self.event_queue.empty(): # we must wait for an valid instruction
                input = self.event_queue.get()
                if input == "kill": # if told to terminate,  do so
                    self.run = False
                    break
                # get the new maps
                self.env_map = input[0]
                self.trace_map = input[1]
                # process them to make the usable versions for rendering
                self.env_colors = self.env_map.reshape(-1,3)
                self.heatmap = self.trace_map.flatten()
                self.heatmap = self.heatmap / self.heatmap.max() * 255

                # draw static features
                pygame.draw.rect(self.screen, self.background_color, self.background)
                pygame.draw.rect(self.screen, self.boundary_color, self.zone_boundary)
                # draw game environment
                for (rect, color) in zip(self.env_rects, self.env_colors):
                    pygame.draw.rect(self.screen, (color[0], color[1], color[2]), rect)

                # draw heatmap
                for (rect, intensity) in zip(self.heatmap_rects, self.heatmap):
                    pygame.draw.rect(self.screen, (intensity, intensity, intensity), rect)

            for event in pygame.event.get(): # should the game be closed, terminate
                if event.type == pygame.QUIT:
                    self.run = False

            pygame.display.update() # update the screen
            pygame.time.wait((1000//self.framerate)) # wait to draw next frame

        pygame.quit()


def launch_viewer(q, name="Explore", x=5, y=20, size=21, window_height=300):
    view = ExploreViewer(q, name, x, y, size, window_height)
    view.start()
