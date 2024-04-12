# fundamental tools for the game environment model
import numpy as np
import os
import pygame
# for visualizations in jupyter
import matplotlib.pyplot as plt


class ExploreGame:
    def __init__(self, size=21):
        # limit size to be between 11 - 101 and odd
        # note that viewer may struggle with large sizes due to grid lines
        if size > 101:
            size = 101
        elif size % 2 == 0:
            size += 1
        if size < 11:
            size = 11
        self.size = int(size)
        self.board = np.zeros((size, size))  # this will be used for obstacles
        self.agent = [int((size - 1) / 2),
                      int((size - 1) / 2)]  # given size must be odd, this places the player in the center
        self.goal = self.get_random_goal()  # puts the goal in a random starting position
        self.won = False  # used for primitive win condition
        self.trace = np.zeros((size, size))  # used for tracking the number of turns spent on each tile in the env
        self.update_trace()  # adds the occurence of spawning in the center to trace

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

    def get_random_goal(self):  # randomly spawns the goal in on the map
        new_x = np.random.choice(self.size)
        new_y = np.random.choice(self.size)
        mid = (self.size - 1) // 2

        if new_x == mid and new_y == mid:  # if the choice would be the middle
            roll = np.random.choice(2)
            list = np.arange(self.size - 1)  # list with one less option
            list[int((self.size - 1) // 2)] = self.size - 1  # replace the midpoint with the outer boundary
            if roll == 1:  # 50/50 chance to reassign either coordinate
                new_x = np.random.choice(list)
            else:
                new_y = np.random.choice(list)
        return [new_x, new_y]

    def soft_reset(self):  # resets the game state but not the trace
        self.board = self.board * 0
        self.agent = [int((self.size - 1) / 2),
                      int((self.size - 1) / 2)]  # given size must be odd, this places the player in the center
        self.goal = self.get_random_goal()
        self.won = False
        self.update_trace()

    def hard_reset(self):  # resets the full object state as if a new instance was made
        self.board = self.board * 0
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