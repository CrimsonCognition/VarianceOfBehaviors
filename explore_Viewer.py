import numpy as np
import math
import random
import itertools
import pygame


class exploreViewer:
    def __init__(self,env_map, trace_map, size=21, framerate=60):
        self.env_map = env_map
        self.trace_map = trace_map

        pygame.init()
        
        self.screen_width = 810
        self.screen_height = 400
        self.framerate = framerate

        self.size = size

        self.grid_width = 6 # keep me even

        self.rect_width = (self.screen_height - (self.size+1)*self.grid_width)/self.size

        self.background = pygame.Rect(0,0,self.screen_width,self.screen_height)
        self.background_color = (32,32, 32)

        self.boundary_width = 10
        self.zone_boundary = pygame.Rect(self.screen_width//2 - self.boundary_width//2, 0, self.boundary_width, self.screen_height)
        self.boundary_color = (255,255, 0)


        self.env_rects = []
        self.env_colors = self.env_map.reshape(-1,3)
        for i in range(self.size**2):
            self.env_rects.append(pygame.Rect((i%self.size)*(self.rect_width+self.grid_width) + self.grid_width, (i//self.size)*(self.rect_width+self.grid_width) + self.grid_width,self.rect_width,self.rect_width))


        self.heatmap_rects = []
        self.heatmap = self.trace_map.flatten()
        self.heatmap = self.heatmap/self.heatmap.max()*255
        for i in range(self.size**2):
            self.heatmap_rects.append(pygame.Rect((i%self.size)*(self.rect_width+self.grid_width) + self.grid_width + self.screen_width//2 + self.boundary_width//2
                                             , (i//self.size)*(self.rect_width+self.grid_width) + self.grid_width,self.rect_width,self.rect_width))

        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))

        self.run = True
    
    def start(self):
        while self.run:
            self.env_colors = self.env_map.reshape(-1,3)
            self.heatmap = self.trace_map.flatten()
            
            pygame.draw.rect(self.screen, self.background_color, self.background)
            pygame.draw.rect(self.screen, self.boundary_color, self.zone_boundary)
            for (rect, color) in zip(self.env_rects, self.env_colors):
                pygame.draw.rect(self.screen, (color[0], color[1], color[2]), rect)

            for (rect, intensity) in zip(self.heatmap_rects, self.heatmap):
                pygame.draw.rect(self.screen, (intensity, intensity, intensity), rect)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.run = False
            
            pygame.display.update()
            pygame.time.wait((1000//self.framerate))

        pygame.quit()