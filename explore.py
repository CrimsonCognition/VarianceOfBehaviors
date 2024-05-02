# fundamental tools for the game environment model
import numpy as np
from scipy.signal import convolve2d
import os
import random
import pygame
from randomchooser import Chooser
# for visualizations in jupyter
import matplotlib.pyplot as plt
import time


class ExploreGame:
    def __init__(self, size=21, maze=True, excluded=True, noise_boundary=4, sparsity=.8):
        # limit size to be between 11 - 101 and odd
        # note that viewer may struggle with large sizes due to grid lines
        if size > 101:
            size = 101
        elif size % 2 == 0:
            size += 1
        if size < 11:
            size = 11

        if noise_boundary < 0:
            noise_boundary = 0
        elif noise_boundary > size//4:  # cap this at a smaller portion of size
            noise_boundary = size//4

        if noise_boundary % 2 == 1:  # The noise_boundary must be even
            noise_boundary += 1

        # util structs
        self.action_pairs = np.array([[-1, 0], [0, -1], [1, 0], [0, 1]], dtype="int8")
        # the action vectors excluding no movement

        self.size = int(size)
        self.center = int(size // 2)
        self.sparsity = sparsity
        self.noise_boundary = noise_boundary
        self.excluded = excluded
        self.maze = maze
        self.board = np.zeros((size, size), dtype="int8")  # this will be used for obstacles
        self.agent = np.array([self.center, self.center], dtype="int8")
        # given size must be odd,
        # this places the player in the center

        self.won = False  # used for win condition
        self.score_goal = 10
        self.score = 0

        self.diag_kernel = np.array([[1, -1], [-1, 1]], dtype="int8")
        if maze:
            self.build_map()

        self.goal_options = None
        self.goal = self.get_random_goal()  # puts the goal in a random starting position

        self.trace = np.zeros((size, size))
        # used for tracking the number of turns spent on each tile in the env
        self.update_trace()  # adds the occurrence of spawning in the center to trace

    def gen_noise_pattern(self):  # generate a noise pattern of spawn points for wall chains
        p = self.sparsity + (.95 - self.sparsity)*(np.random.random() + np.random.random())/2
        # Binomial distribution between sparsity and .95
        q = 1 - p  # The probability of having a wall root
        q /= 10  # splitting q to be distributed between multiple possibilities
        offset = self.noise_boundary  # must be even - used to prevent walls from starting on the edges
        # Note this does not prevent walls from propagating to the edges
        pattern = np.random.choice([0, 9, 6, 4], (self.size-offset, self.size-offset), p=[p, q, 3*q, 6*q])
        full_pattern = np.zeros((self.size, self.size))
        offset = offset // 2
        full_pattern[offset:self.size-offset, offset:self.size - offset] = pattern
        return full_pattern

    def action_switch(self, coord, action):  # returns the next coord given a current coord and action
        return coord - self.action_pairs[action]

    def propagate_wall(self, coord, val, last_move):
        # takes a root location, and build val number of
        # leaves recursively, does not allow for doubling back with last_move and detection

        if val == 0:  # The base case, if there are no walls left to make, end
            return True
        else:
            probs = np.arange(4)  # uniform if no chooser
            if last_move < 4:
                probs[(last_move + 2) % 4] = -1  # disallow going backwards
            probs = probs[probs >= 0]
            action = probs[np.random.randint(0, probs.size)]

            target = self.action_switch(coord, action)  # use its choice - as it may include the backwards
            # stay in bounds
            target[0] = max(0, min(target[0], self.size - 1))
            target[1] = max(0, min(target[1], self.size - 1))

            self.board[coord[0]][coord[1]] = 1  # place a wall
            self.propagate_wall(target, val-1, action)  # Recur

    def clean_diagonal_walls(self, p=.5):
        # We want to remove instances of obstructions composed of diagonal walls
        # We are going to convolve looking for strictly diagonal obstructions
        # Specifically we are looking for wall layouts that look like
        # [1, 0]  or [0, 1]
        # [0, 1]     [1, 0]
        # We want to correct or remove these as they prevent passage via cardinal motion but are not cardinally
        # connected themselves.
        convolution = convolve2d(self.board, self.diag_kernel, mode="valid").astype("int8")
        mask = (convolution == 2) | (convolution == -2)
        if mask.sum() == 0:
            return True
        diagonals_1, diagonals_2 = np.where(mask)
        length = diagonals_1.shape[0]
        diag1 = np.zeros((length, 2), dtype="int8")
        diag1[:, 0] = diagonals_1
        diag1[:, 1] = diagonals_2

        diag2 = diag1 + [1, 1]
        adiag1 = diag1 + [0, 1]
        adiag2 = diag1 + [1, 0]

        diag_mask = (convolution[diagonals_1, diagonals_2] == 2).reshape(-1, 1)
        adiag_mask = (convolution[diagonals_1, diagonals_2] == -2).reshape(-1, 1)

        select = np.random.choice([True, False], (length, 1))

        choices = diag1 * select + diag2 * (1 - select)
        anti_choices = adiag1 * select + adiag2 * (1 - select)

        temp = choices * diag_mask + anti_choices * adiag_mask
        temp2 = anti_choices * diag_mask + choices * adiag_mask

        choices = temp.copy()
        anti_choices = temp2.copy()

        add_or_remove = np.random.choice([False, True], (length, 1), p=[1 - p, p])

        result = (anti_choices * add_or_remove + choices * (1 - add_or_remove)).astype("int8")

        result, idx = np.unique(result, True, axis=0)

        result = np.hstack([result, add_or_remove[idx]]).astype("int8")

        self.board[result[:, 0], result[:, 1]] = result[:, 2]
        return self.clean_diagonal_walls(max(0, p - .1))

    def fill_map_holes(self):  # special case filling on map
        #  this function fills any single tile that has a wall in each cardinal direction but is not a wall itself
        #  this function should take place after diagonals are cleaned
        # In other words, if we find closed regions of size 1, fill them in, rather than include them for reconnection
        coords = np.where(self.board == 0)  # for all open tiles
        for (i, j) in zip(coords[0], coords[1]):
            neighbors = []
            for k in range(4):  # prep neighbor coords
                neighbors.append(self.action_switch((i, j), k))
            walled = 0
            for tile in neighbors:  # if inside the map bounds check the neighbor value on board
                if 0 <= tile[0] < self.size and 0 <= tile[1] < self.size:
                    walled += self.board[tile[0]][tile[1]]
                else:  # otherwise count the bounds as a wall
                    walled += 1
            if walled == 4:  # if we are blocked off / are an island
                self.board[i][j] = 1  # become a wall as well

    def build_map(self):  # Procedurally generates a map
        #t1 = time.time_ns()
        pattern = self.gen_noise_pattern()  # we need a noise pattern to build from

        # For all tiles that have a root, call propagate wall
        root_i, root_j = np.where(pattern > 0)
        vals = pattern[root_i, root_j]
        #t2 = time.time_ns()
        for (i, j, v) in zip(root_i, root_j, vals):
            # a function that takes i, j, val to build a stem
            self.propagate_wall(np.array([i, j]), v, 4)  # last move is 4 to indicate no "last" move exists yet
        #t3 = time.time_ns()

        self.clean_diagonal_walls()  # quick pass for initial clean up
        self.fill_map_holes()  # remove trivial cases of map defects
        self.board[self.center, self.center] = 0  # Force spawn point to be open
        self.clean_diagonal_walls(0)  # just in case the above correction created a diagonal, is destructive only

        #t4 = time.time_ns()
        cleaned, sample = self.correct_map_defects()  # This function connects all closed regions to spaw point
        #t5 = time.time_ns()

        #print("P1", (t2-t1)/1000000)
        #print("P2", (t3-t2)/1000000)
        #print("P3", (t4-t3)/1000000)
        #print("P4", (t5-t4)/1000000)
        #print("Total", (t5-t1)/1000000)

        return pattern, cleaned

    def update_trace(self):  # increments the tile in the trace that the agent is currently on.
        self.trace[self.agent[0], self.agent[1]] += 1

    def get_trace(self):
        return self.trace.copy()

    def propagate_from_spawn(self, board):
        coord_list = []
        marker = -1
        coord_list.append(self.agent.reshape(1, 2))
        board[self.agent[0], self.agent[1]] = marker
        # Starting from the spawn point, mark each neighboring free space as reachable,
        # then check those space's neighbors until all reachable spaces from spawn have been checked
        # then return this marked down board, where reachable tiles are noted as the marker value
        # walls remain as 1s and closed regions of unreachable space are 0s
        while coord_list:  # While we have coords to check
            coord = coord_list.pop()
            neighbors = np.zeros((4 * coord.shape[0], 2), dtype="int")
            for (i, c) in zip(range(coord.shape[0]), coord):
                temp = self.action_pairs + c
                neighbors[4 * i:4 * (i + 1), :] = temp

            neighbors = np.unique(neighbors, axis=0)
            # remove out of bounds
            neighbors = neighbors[(neighbors.min(axis=1) >= 0) & (neighbors.max(axis=1) < self.size), :]

            neighbors = neighbors[board[neighbors[:, 0], neighbors[:, 1]] == 0]
            board[neighbors[:, 0], neighbors[:, 1]] = marker
            if neighbors.shape[0] > 0:
                coord_list.append(neighbors)
        return board

    def map_is_valid(self):  # returns true if every "open" tile is reachable from spawn point
        test = self.propagate_from_spawn(self.board.copy())
        return (test == 0).sum() == 0, test

    def find_walls_touching_target(self, sample, target):
        walls_i, walls_j = np.where(sample == 1)
        indxs = np.vstack([walls_i, walls_j], dtype="int8").T
        up = indxs + self.action_pairs[0]
        t = (up.min(axis=1) >= 0) & (up.max(axis=1) < self.size)
        up = up[t]
        up = up[sample[up[:, 0], up[:, 1]] == target]
        up = up - self.action_pairs[0]

        left = indxs + self.action_pairs[1]
        t = (left.min(axis=1) >= 0) & (left.max(axis=1) < self.size)
        left = left[t]
        left = left[sample[left[:, 0], left[:, 1]] == target]
        left = left - self.action_pairs[1]

        down = indxs + self.action_pairs[2]
        t = (down.min(axis=1) >= 0) & (down.max(axis=1) < self.size)
        down = down[t]
        down = down[sample[down[:, 0], down[:, 1]] == target]
        down = down - self.action_pairs[2]

        right = indxs + self.action_pairs[3]
        t = (right.min(axis=1) >= 0) & (right.max(axis=1) < self.size)
        right = right[t]
        right = right[sample[right[:, 0], right[:, 1]] == target]
        right = right - self.action_pairs[3]

        out = np.vstack([up, left, down, right], dtype="int8")
        out = np.unique(out, axis=0)
        return out

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

            collisions = np.vstack([walls_on_reachable, walls_on_enclosed])
            uniques, u, cs = np.unique(collisions, True, False, True, axis=0)
            collisions = collisions[u[cs > 1]]

            temp2 = []
            temp3 = []

            for x in walls_on_enclosed:
                temp2.append([x[0], x[1]])
            for x in collisions:
                temp3.append([x[0], x[1]])

            walls_on_enclosed = temp2
            collisions = temp3

            if collisions:
                collisions = np.array(collisions, dtype="int16")
                self.board[collisions[:, 0], collisions[:, 1]] = 0
                self.clean_diagonal_walls(0)
                return self.correct_map_defects()
            else:
                # decimate enclosing walls
                walls_on_enclosed = np.array(walls_on_enclosed)
                length = walls_on_enclosed.shape[0]
                decimation_factor = 20
                # by increasing decimation rate for larger maps, we may sacrifice some quality to achieve
                # a close to constant number of iterations to resolve the map
                if self.size >= 81:
                    decimation_factor = 5
                elif self.size >= 41:
                    decimation_factor = 10
                val = max(1, length // decimation_factor)
                indxs = np.random.choice(np.arange(length), val, False)
                self.board[walls_on_enclosed[indxs, 0], walls_on_enclosed[indxs, 1]] = 0
                self.clean_diagonal_walls(0)
                return self.correct_map_defects()

    def on_goal_reached(self):
        self.score += 1
        if self.score >= self.score_goal:
            self.won = True
        else:
            self.new_map()

    def update(self, action):  # the game logic on a turn execution
        if self.won:
            return self.won
        old = self.agent.copy()  # this will be used for collisions later
        if action < 4:
            coord = self.action_switch(self.agent, action)
            self.agent = np.array([min(self.size-1, max(0, coord[0])), min(self.size-1, max(0, coord[1]))])
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
        delta = int(self.size // 4)
        lower = self.center - delta
        upper = self.center + delta
        for (i, j) in zip(free_squares[0], free_squares[1]):
            if i == self.center and j == self.center:  # Skip where the agent must spawn,
                # yes this is redundant if exclusion is on
                continue
            elif self.excluded and (lower <= i <= upper) and (lower <= j <= upper):
                continue
            else:
                self.goal_options.append(np.array([i, j]))

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
        self.agent = np.array([self.center, self.center])
        # given size must be odd, this places the player in the center

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
