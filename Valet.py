#!/usr/bin/env python3
import numpy as np
import math

import matplotlib.pyplot as plt

from Env import Map, Map_Trailer
import vehicle_models
from Hybrid_astar import*
from Hybrid_astar_trailer import*

ENV_WIDTH = 41
ENV_HEIGHT = 41
ARROW_LEN = 3


'''
Running the Hybrid A* planner for all three vehicles
'''
 
def main():


 for VEHICLE_TYPE in  ["diffdrive","car","trucktrailer"]:
    start_pose = [3.0, 35.0, np.deg2rad(0)]
    goal_pose = [21.0, 5.0, np.deg2rad(180)]
    if VEHICLE_TYPE=="trucktrailer":
        start_pose = [10.0, 35.0, np.deg2rad(0), np.deg2rad(0)]
        goal_pose = [20.0, 5.0, np.deg2rad(0), np.deg2rad(-30)]
        

    if VEHICLE_TYPE == "diffdrive":
        map = Map(ENV_WIDTH, ENV_HEIGHT)
        map.buildEnv(VEHICLE_TYPE)
        map_env = calculateEnvParameters(map.envXCoord, map.envYCoord, 4, np.deg2rad(15.0))
        x, y, yaw = hybridAstar(start_pose, goal_pose, map_env, plt)

    if VEHICLE_TYPE=="car":
        map = Map(ENV_WIDTH, ENV_HEIGHT)
        map.buildEnv(VEHICLE_TYPE)
        map_env = calculateEnvParameters(map.envXCoord, map.envYCoord, 4, np.deg2rad(15.0))
        x, y, yaw = hybridAstar(start_pose, goal_pose, map_env, plt)

    if VEHICLE_TYPE=="trucktrailer":
        map = Map_Trailer(ENV_WIDTH, ENV_HEIGHT)
        map.buildEnv(VEHICLE_TYPE)
        map_env = calculateEnvParameters(map.envXCoord, map.envYCoord, 4, np.deg2rad(15.0))
        x, y, yaw, yaw_t, directions = hybridAstar_trailer(start_pose, goal_pose, map_env, plt)
    
    mng = plt.get_current_fig_manager()
 
 

    for k in range(len(x)):
        plt.cla()
        plt.xlim(min(map.envXCoord), max(map.envXCoord)) 
        plt.ylim(min(map.envYCoord), max(map.envYCoord))

    
        if VEHICLE_TYPE=="trucktrailer":            
            vehicle_models.drawTruck(x[k], y[k], yaw[k], yaw_t[k],  'black', )
            vehicle_models.drawTruck(start_pose[0], start_pose[1], start_pose[2], start_pose[3], 'green')
            vehicle_models.drawTruck(goal_pose[0], goal_pose[1], goal_pose[2],  goal_pose[3],'red')

        if VEHICLE_TYPE=="diffdrive":
            vehicle_models.drawDiffDrive(start_pose[0], start_pose[1], start_pose[2], 'green')
            vehicle_models.drawDiffDrive(goal_pose[0], goal_pose[1], goal_pose[2], 'red')
            vehicle_models.drawDiffDrive(x[k], y[k], yaw[k], 'grey' )

        if VEHICLE_TYPE=="car":
            vehicle_models.drawCar(start_pose[0], start_pose[1], start_pose[2], 'green')
            vehicle_models.drawCar(goal_pose[0], goal_pose[1], goal_pose[2], 'red')
            vehicle_models.drawCar(x[k], y[k], yaw[k], 'grey' )

        plt.arrow(x[k],y[k],  ARROW_LEN*math.cos(yaw[k]),  ARROW_LEN*math.sin(yaw[k]), width=0.05)

        plt.plot(map.envXCoord, map.envYCoord, "sk")
        plt.plot(x, y, linewidth=1.5, color='r', zorder=1)
        plt.axis('equal')
        plt.title(str(VEHICLE_TYPE))
        plt.pause(0.1)
        
    plt.show()




if __name__ == '__main__':
    main()
