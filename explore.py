# fundamental tools for the game environment model
import numpy as np
import os
import random
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
        self.agent = [int((size - 1) / 2),
                      int((size - 1) / 2)]  # given size must be odd, this places the player in the center
        if maze:
            self.build_map()
        self.goal_options = None
        self.goal = self.get_random_goal()  # puts the goal in a random starting position
        self.won = False  # used for primitive win condition
        self.score_goal = 10
        self.score = 0
        self.trace = np.zeros((size, size))  # used for tracking the number of turns spent on each tile in the env
        self.update_trace()  # adds the occurence of spawning in the center to trace

    def gen_noise_pattern(self):  # generate a noise pattern of spawn points for wall chains
        p = .7 + (.95 - .7)*(np.random.random() + np.random.random())/2  # Binomial distribution between .7 and .95
        q = 1 - p
        q /= 10
        offset = 4  # must be even
        pattern = np.random.choice([0, 9, 6, 4], (self.size-offset, self.size-offset), p=[p, q, 3*q, 6*q])
        full_pattern = np.zeros((self.size, self.size))
        full_pattern[2:self.size-offset//2, 2:self.size - offset//2] = pattern
        return full_pattern

    def action_switch(self, coord, action):
        return coord - self.action_pairs[action]

    def propagate_wall(self, coord, val, last_move, chooser=None):  # takes a root location, and build val number of
        # leaves recursively, does not allow for doubling back with last_move and detection
        if val == 0:
            return True
        else:
            probs = np.ones(4)
            if chooser is not None:
                probs = chooser.probs.copy()
            if last_move < 4:
                probs[(last_move + 2) % 4] = 0  # disallow going backwards
            probs = probs/probs.sum()  # normalize
            action = np.random.choice([0, 1, 2, 3], p=probs)
            chooser.get_random_action()  # iterates the chooser but doesn't
            target = self.action_switch(coord, action)  # use its choice - as it may include the backwards
            # stay in bounds
            target[0] = max(0, min(target[0], 20))
            target[1] = max(0, min(target[1], 20))

            if self.board[target[0]][target[1]] == 0:  # if the space does not have an obstacle yet
                self.board[coord[0]][coord[1]] = 1  # place a wall
            self.propagate_wall(target, val-1, action, chooser)  # Recurr

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

    def clean_diagonal_walls(self, p=1):  # We want to remove instances of obstructions composed of diagonal walls
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

    def fill_map_holes(self): # special case filling on map
        #  this function fills any single tile that has a wall in each cardinal direction but is not a wall itself
        #  this function should take place after diagonals are cleaned
        coords = np.where(self.board == 0)  # for all open tiles
        for (i, j) in zip(coords[0], coords[1]):
            neighbors = []
            for k in range(4): # prep neighbor coords
                neighbors.append(self.action_switch((i, j), k))
            walled = 0
            for tile in neighbors:  # if inside the map bounds check the neighbor value on board
                if tile[0] >= 0 and tile[0] < self.size and tile[1] >= 0 and tile[1] < self.size:
                    walled += self.board[tile[0]][tile[1]]
                else:  # otherwise count the bounds as a wall
                    walled += 1
            if walled == 4:  # if we are blocked off / are an island
                self.board[i][j] = 1  # become a wall as well

    def build_map(self):
        pattern = self.gen_noise_pattern()  # we need a noise pattern to build from
        chooser = Chooser(4, [5, 15, 10, 5], 3)
        # let's do a naive search over the pattern, more efficient solutions exist in numpy but this is legible
        for i in range(self.size):
            for k in range(self.size):
                if pattern[i][k] > 0:
                    val = pattern[i][k]
                    # a function that takes i, k, val to build a stem
                    self.propagate_wall((i, k), val, 4, chooser)
        cleaned = self.clean_diagonal_walls()
        self.fill_map_holes()
        center = int((self.size - 1) / 2)
        self.board[center, center] = 0  # Force spawn point to be open
        self.clean_diagonal_walls(0)
        cleaned, sample = self.correct_map_defects()
        return pattern, cleaned

    def update_trace(self):  # increments the tile in the trace that the agent is currently on.
        self.trace[self.agent[0], self.agent[1]] += 1

    def get_trace(self):
        return self.trace.copy()

    def propagate_from_spawn(self, board): #
        coord_list = []
        marker = -1
        coord_list.append(self.agent)
        board[self.agent[0], self.agent[1]] = marker
        while coord_list:
            coord = coord_list.pop()
            neighbors = []
            for k in range(4):  # prep neighbor coords
                neighbors.append(self.action_switch((coord[0], coord[1]), k))
            for neighbor in neighbors:
                if neighbor[0] >= 0 and neighbor[0] < self.size and neighbor[1] >= 0 and neighbor[1] < self.size:
                    if board[neighbor[0], neighbor[1]] == 0:
                        coord_list.append(neighbor)
                        board[neighbor[0], neighbor[1]] = marker
        return board

    def map_is_valid(self): # returns true if every "open" tile is reachable from spawn point
        test = self.propagate_from_spawn(self.board.copy())
        return (test == 0).sum() == 0, test

    def find_walls_touching_target(self, sample, target):
        walls = np.where(sample == 1)
        walls_touching_target = []
        for (i, j) in zip(walls[0], walls[1]): # for all walls
            for k in range(4):
                coord = self.action_switch((i, j), k)  # find walls that neighbor reachables
                if coord[0] < 0 or coord[0] >= self.size or coord[1] < 0 or coord[1] >= self.size:
                    continue
                elif sample[coord[0], coord[1]] == target:
                    walls_touching_target.append([i, j])
                    break
        return walls_touching_target

    def are_neighbors(self, coord_1, coord_2):
        for k in range(4):
            act = self.action_switch(coord_1, k)
            if coord_2 == [act[0], act[1]]:
                return True
        return False

    def build_neighborhood(self, group, pool):
        if group:
            out = []
            # find neighbors of group
            for g in group:
                for k in range(4):
                    for p in range(len(pool)):
                        if self.are_neighbors(g, pool[p]):
                            out.append(pool.pop(p))
                            break
            out_recurr = self.build_neighborhood(out, pool)  # find the neighbors neighbors
            final = []
            for x in group:
                final.append(x)
            for x in out_recurr:
                final.append(x)
            return final
        else:
            return []

    def build_neighborhoods(self, coords):
        neighborhoods = []
        while coords:
            neighborhoods.append([coords.pop(0)])
            neighborhoods[-1] = self.build_neighborhood(neighborhoods[-1], coords)
        return neighborhoods

    def correct_map_defects(self):
        validity, sample = self.map_is_valid()
        if validity:
            return validity, sample
        else:
            walls_on_reachable = self.find_walls_touching_target(sample, -1)
            walls_on_enclosed = self.find_walls_touching_target(sample, 0)
            collisions = []
            for e in walls_on_enclosed:
                for r in walls_on_reachable:
                    if r == e:
                        collisions.append(r)
            random.shuffle(collisions)

            if collisions:
                neighborhoods = self.build_neighborhoods(collisions)
                random.shuffle(neighborhoods)
                for set in neighborhoods:
                    random.shuffle(set)
                    self.board[set[0][0], set[0][1]] = 0
                    validity, sample = self.map_is_valid()
                    if validity:
                        break
                self.clean_diagonal_walls(0)
                if not validity:
                    return self.correct_map_defects()
            else:
                neighborhoods = self.build_neighborhoods(walls_on_enclosed)
                random.shuffle(neighborhoods)
                for set in neighborhoods:
                    random.shuffle(set)
                    self.board[set[0][0], set[0][1]] = 0
                    validity, sample = self.map_is_valid()
                    if validity:
                        break
                self.clean_diagonal_walls(0)
                if not validity:
                    return self.correct_map_defects()
            return self.map_is_valid()

    def on_goal_reached(self):
        self.score += 1
        if self.score >= self.score_goal:
            self.won = True
        else:
            self.new_map()

    def update(self, action):  # the game logic on a turn execution
        if self.won:
            return self.won
        old = [self.agent[0], self.agent[1]]  # this will be used for collisions later
        if action < 4:
            coord = self.action_switch(self.agent, action)
            self.agent = [min(self.size-1, max(0, coord[0])), min(self.size-1, max(0, coord[1]))]
        else:
            return self.won

        if self.board[self.agent[0], self.agent[1]] == 1:  # collide with wall = don't move
            self.agent = old
        elif self.agent[0] == self.goal[0] and self.agent[1] == self.goal[1]:  # if we hit the goal, add points
            self.on_goal_reached()
            # we will want to add new goals for later
        # if action is 0 = do nothing
        self.update_trace()
        return self.won

    def build_goal_options(self):
        self.goal_options = []
        free_squares = np.where(self.board == 0)
        center = int((self.size - 1) / 2)
        delta = int(self.size // 4)
        lower = center - delta
        upper = center + delta
        for (i, j) in zip(free_squares[0], free_squares[1]):
            if i == center and j == center:  # Skip where the agent must spawn, yes this is redundant if exclusion is on
                continue
            elif (lower <= i <= upper) and (lower <= j <= upper):
                continue
            else:
                self.goal_options.append([i, j])

    def get_random_goal(self):  # randomly spawns the goal in on the map
        if self.goal_options is None:
            self.build_goal_options()

        choice = np.random.choice(np.arange(len(self.goal_options)))

        return self.goal_options[choice]

    def new_map(self, build=True):
        if build and self.maze:
            self.board = self.board * 0
            self.build_map()
        self.goal_options = None
        self.agent = [int((self.size - 1) / 2),
                      int((self.size - 1) / 2)]  # given size must be odd, this places the player in the center
        self.goal = self.get_random_goal()
        self.update_trace()

    def soft_reset(self, rebuild=True):  # resets the game state but not the trace and optionally the map
        self.new_map(rebuild)
        self.won = False
        self.score = 0

    def hard_reset(self):  # resets the full object state as if a new instance was made
        self.trace = self.trace * 0
        self.new_map()
        self.won = False
        self.score = 0

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
