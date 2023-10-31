#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import math

'''
Building the environment with obstacles
'''
class Map:
    def __init__(self,width, height) -> None:
        self.width = width
        self.height = height
        self.envXCoord, self.envYCoord = [], []
        self.left_obs_anchor  = [self.width*0.12, self.width*0.05]
        self.right_obs_anchor  = [self.width*0.65, self.width*0.05]
        self.center_obs_anchor = [self.width*0.45, self.width*0.5]
        self.obs_width =  self.width*0.20
        self.obs_height =  self.width*0.1
        self.obs_delta =  self.width*0.25

    def buildEnv(self, vehicle_type):
        
        for row in range(self.height):
            for col in range(self.width):

               
                if row == 0:
                    self.envXCoord.append(col)
                    self.envYCoord.append(0)

                if col == 0:
                    self.envXCoord.append(0)
                    self.envYCoord.append(row)

                if row == (self.height-1):
                    self.envXCoord.append(col)
                    self.envYCoord.append(self.height-1)

                if col == (self.width-1):
                    self.envXCoord.append(self.width-1)
                    self.envYCoord.append(row)

                if col >= self.right_obs_anchor[0] and col <= (self.right_obs_anchor[0] + self.obs_width ):
                    if row >= self.right_obs_anchor[1] and row <= (self.right_obs_anchor[1] + self.obs_height):
                        self.envXCoord.append(col)
                        self.envYCoord.append(row)

                if col >= self.center_obs_anchor[0] and col <= (self.center_obs_anchor[0] + self.obs_width + self.width*0.20):
                    if row >= self.center_obs_anchor[1] and row <= (self.center_obs_anchor[1] + self.obs_height +self.obs_delta):
                        self.envXCoord.append(col)
                        self.envYCoord.append(row)
                        
                if vehicle_type == "car" or "diffdrive":
                    #Plotting left obstacle
                    if col >= self.left_obs_anchor[0] and col <= (self.left_obs_anchor[0] + self.obs_width):
                        if row >= self.left_obs_anchor[1] and row <= (self.left_obs_anchor[1] + self.obs_height):
                            self.envXCoord.append(col)
                            self.envYCoord.append(row)
                            
                            
class Map_Trailer:
    def __init__(self,width, height) -> None:
        self.width = width
        self.height = height
        self.envXCoord, self.envYCoord = [], []
        self.left_obs_anchor  = [self.width*0.05, self.width*0.05]
        self.right_obs_anchor  = [self.width*0.65, self.width*0.05]
        self.center_obs_anchor = [self.width*0.45, self.width*0.5]
        self.obs_width =  self.width*0.20
        self.obs_height =  self.width*0.1
        self.obs_delta =  self.width*0.25

    def buildEnv(self, vehicle_type):
        
        for row in range(self.height):
            for col in range(self.width):

               
                if row == 0:
                    self.envXCoord.append(col)
                    self.envYCoord.append(0)

                if col == 0:
                    self.envXCoord.append(0)
                    self.envYCoord.append(row)

                if row == (self.height-1):
                    self.envXCoord.append(col)
                    self.envYCoord.append(self.height-1)

                if col == (self.width-1):
                    self.envXCoord.append(self.width-1)
                    self.envYCoord.append(row)

                if col >= self.right_obs_anchor[0] and col <= (self.right_obs_anchor[0] + self.obs_width ):
                    if row >= self.right_obs_anchor[1] and row <= (self.right_obs_anchor[1] + self.obs_height):
                        self.envXCoord.append(col)
                        self.envYCoord.append(row)

                if col >= self.center_obs_anchor[0] and col <= (self.center_obs_anchor[0] + self.obs_width + self.width*0.20):
                    if row >= self.center_obs_anchor[1] and row <= (self.center_obs_anchor[1] + self.obs_height +self.obs_delta):
                        self.envXCoord.append(col)
                        self.envYCoord.append(row)

                                   #Plotting left obstacle
                if col >= self.left_obs_anchor[0] and col <= (self.left_obs_anchor[0] + self.obs_width):
                        if row >= self.left_obs_anchor[1] and row <= (self.left_obs_anchor[1] + self.obs_height):
                            self.envXCoord.append(col)
                            self.envYCoord.append(row)
                            
        
                   
            
