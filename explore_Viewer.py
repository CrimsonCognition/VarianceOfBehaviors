import numpy as np
import math
import random
import itertools
import pygame


class exploreViewer:
    def __init__(self, events, name="Explore", size=21, framerate=60):
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
        self.screen_width = 810
        self.screen_height = 400

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