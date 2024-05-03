import sys
import time
sys.path.append('../Kinematics')
sys.path.append('../RL')

import gym
import numpy as np 
import math 
from gym import spaces 
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation 


from forward_kinematics import three_section_robot, jacobian_matrix
from forward_kinematics import trans_mat_cc, couple_transformations
from AmorphousSpace import AmorphousSpace

class continuumEnv(gym.Env):
    def __init__(self): # TODO: Add some of the reqired attributes as optional parameter such as __init__(self, delta_kappa = 0.001)
        self.delta_kappa = 0.001     # necessary for the numerical differentiation
        self.kappa_dot_max = 1.000  # max derivative of curvature
        self.kappa_max = 16.00      # max curvature for the robot
        self.kappa_min = -4.00      # min curvature for the robot
        l1 = 0.1000;                # first segment of the robot in meters
        l2 = 0.1000;                # second segment of the robot in meters
        l3 = 0.1000;                # third segment of the robot in meters
        self.stop = 0               # variable to make robot not move after exeeding max, min general kappa value
        self.l = [l1, l2, l3]       # stores the length of each segment of the robot
        self.dt =  5e-2             # sample sizes
        self.J = np.zeros((2,3))    # initializes the Jacobian matrix  
        self.error = 0              # initializes the error
        self.previous_error = 0     # initializes the previous error
        self.start_kappa = [0,0,0]  # initializes the start kappas for the three segments
        self.time = 0               # to count the time of the simulation
        self.overshoot0 = 0
        self.overshoot1 = 0
        self.position_dic = {'Section1': {'x':[],'y':[]}, 'Section2': {'x':[],'y':[]}, 'Section3': {'x':[],'y':[]}}
        # Define the observation and action space from OpenAI Gym
        high = np.array([0.2, 0.3, 0.16, 0.3], dtype=np.float32) # [0.16, 0.3, 0.16, 0.3]
        low  = np.array([-0.3, -0.15, -0.27, -0.11], dtype=np.float32) # [-0.27, -0.11, -0.27, -0.11]
        self.action_space = spaces.Box(low=-1*self.kappa_dot_max, high=self.kappa_dot_max,shape=(3,), dtype=np.float32)
        self.observation_space = AmorphousSpace()
        
    def step_error_comparison(self,u): # reward is -1.00 or -0.50 or 1.00
        
        x,y,goal_x,goal_y = self.state # Get the current state as x,y,goal_x,goal_y
        global new_x 
        global new_y
        global new_goal_x
        global new_goal_y

        dt =  self.dt # Time step
        
        u = np.clip(u, -self.kappa_dot_max, self.kappa_dot_max) # Clip the input to the range of the -1,1
        
        self.error = math.sqrt(((goal_x-x)**2)+((goal_y-y)**2)) # Calculate the error squared
        
        # Just to show if the robot is moving along the goal or not
        if self.error < self.previous_error:
            #self.costs -= 1
            # UNCOMMENT HERE !!!!!!!
            pass
            # print("=========================POSITIVE MOVE=========================")
        
        if self.error < self.previous_error:
            self.costs = 1.00
        elif self.error == self.previous_error:
            self.costs = -0.50
        else:
            self.costs = -1.0
        
        self.previous_error = self.error
        
        # if the error is less than 0.01, the robot is close to the goal and returns done
        if self.error <= 0.010:
            done = True
        else :
            done = False
              
        # This if and else statement is to avoid the robot to move if the kappas are at the limits
        if self.stop == 0:
            self.J = jacobian_matrix(self.delta_kappa, self.kappa1, self.kappa2, self.kappa3, self.l)
            x_vel = self.J @ u
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
            
        elif self.stop == 1:
            self.J = jacobian_matrix(self.delta_kappa, self.kappa1, self.kappa2, self.kappa3, self.l)
            x_vel = self.J @ np.append([0],u[1:3])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
        
        elif self.stop == 2:
            self.J = jacobian_matrix(self.delta_kappa, self.kappa1, self.kappa2, self.kappa3, self.l)
            x_vel = self.J @ np.append(np.append(u[0],[0]),u[2])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
            
        elif self.stop == 3:
            self.J = jacobian_matrix(self.delta_kappa, self.kappa1, self.kappa2, self.kappa3, self.l)
            x_vel = self.J @ np.append(u[0:2],[0])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
            
        elif self.stop == 4:
            self.J = jacobian_matrix(self.delta_kappa, self.kappa1, self.kappa2, self.kappa3, self.l)
            x_vel = self.J @ np.append([0,0],u[2])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
        
        elif self.stop == 5:
            self.J = jacobian_matrix(self.delta_kappa, self.kappa1, self.kappa2, self.kappa3, self.l)
            x_vel = self.J @  np.append(np.append([0],u[1]),[0])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
        
        elif self.stop == 6:
            self.J = jacobian_matrix(self.delta_kappa, self.kappa1, self.kappa2, self.kappa3, self.l)
            x_vel = self.J @ np.append(u[0],[0,0])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
            
        elif self.stop == 7:
            pass

        
        # Update the curvatures
        self.kappa1 += u[0] * dt 
        self.kappa2 += u[1] * dt
        self.kappa3 += u[2] * dt

        # TODO -> Solve the situation when kappas are zero in Homogenous matrix
        # Maybe when it is Zero try except and Raise an error
        self.kappa1 = np.clip(self.kappa1, self.kappa_min, self.kappa_max)
        self.kappa2 = np.clip(self.kappa2, self.kappa_min, self.kappa_max)
        self.kappa3 = np.clip(self.kappa3, self.kappa_min, self.kappa_max)

        # To check which curvature value are at the limits
        self.stop = 0
        k1 = self.kappa1 <= self.kappa_min or self.kappa1 >= self.kappa_max
        k2 = self.kappa2 <= self.kappa_min or self.kappa2 >= self.kappa_max
        k3 = self.kappa3 <= self.kappa_min or self.kappa3 >= self.kappa_max
        
        if k1:
            self.stop = 1          
        elif k2:
            self.stop = 2            
        elif k3:
            self.stop = 3     
        if k1 and k2:
            self.stop = 4     
        elif k1 and k3:
            self.stop = 5     
        elif k2 and k3:
            self.stop = 6          
        if k1 and k2 and k3:
            self.stop = 7     
        if self.observation_space.contains([new_x, new_y]):
            pass
        else:
            # Clip the states to avoid the robot to go out of the workspace
            self.overshoot0 += 1
            #print(new_x, new_y)
            new_x, new_y = self.observation_space.clip([new_x,new_y])
            #print(new_x, new_y)
            # TODO: When it is clipped, then write a algorithm to fill the empy trajectory between before clip and after clip

        if self.observation_space.contains([goal_x, goal_y]):
            new_goal_x, new_goal_y = goal_x, goal_y
        else:
            # Clip the states to avoid the robot to go out of the workspace
            self.overshoot1 += 1
            #print(goal_x,goal_y)
            new_goal_x, new_goal_y = self.observation_space.clip([goal_x,goal_y])
       
        # States of the robot in numpy array
        self.state = np.array([new_x,new_y,new_goal_x,new_goal_y])
        
        return self._get_obs(), self.costs, done, {} # Return the observation, the reward (-costs) and the done flag

    def step_minus_euclidean_square(self,u): # reward is -(e^2)
        
        x,y,goal_x,goal_y = self.state # Get the current state as x,y,goal_x,goal_y
        
        # global variables to be used in the reward function
        global new_x 
        global new_y
        global new_goal_x
        global new_goal_y
        dt =  self.dt # Time step
        
        u = np.clip(u, -self.kappa_dot_max, self.kappa_dot_max) # Clip the input to the range of the -1,1
        
        self.error = ((goal_x-x)**2)+((goal_y-y)**2) # Calculate the error squared
        self.costs = self.error # Set the cost (reward) to the error squared
        
        # Just to show if the robot is moving along the goal or not
        if self.error < self.previous_error:
            #self.costs -= 1
            # UNCOMMENT HERE !!!!!!!
            pass
            # print("=========================POSITIVE MOVE=========================")
            
        
        self.previous_error = self.error # Update the previous error

        if math.sqrt(self.costs) <= 0.01:
            done = True
        else :
            done = False     
        # This if and else statement is to avoid the robot to move if the kappas are at the limits
        if self.stop == 0:
            self.J = jacobian_matrix(self.delta_kappa, self.kappa1, self.kappa2, self.kappa3, self.l)
            x_vel = self.J @ u
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
            
        elif self.stop == 1:
            self.J = jacobian_matrix(self.delta_kappa, self.kappa1, self.kappa2, self.kappa3, self.l)
            x_vel = self.J @ np.append([0],u[1:3])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
        
        elif self.stop == 2:
            self.J = jacobian_matrix(self.delta_kappa, self.kappa1, self.kappa2, self.kappa3, self.l)
            x_vel = self.J @ np.append(np.append(u[0],[0]),u[2])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
            
        elif self.stop == 3:
            self.J = jacobian_matrix(self.delta_kappa, self.kappa1, self.kappa2, self.kappa3, self.l)
            x_vel = self.J @ np.append(u[0:2],[0])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
            
        elif self.stop == 4:
            self.J = jacobian_matrix(self.delta_kappa, self.kappa1, self.kappa2, self.kappa3, self.l)
            x_vel = self.J @ np.append([0,0],u[2])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
        
        elif self.stop == 5:
            self.J = jacobian_matrix(self.delta_kappa, self.kappa1, self.kappa2, self.kappa3, self.l)
            x_vel = self.J @  np.append(np.append([0],u[1]),[0])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
        
        elif self.stop == 6:
            self.J = jacobian_matrix(self.delta_kappa, self.kappa1, self.kappa2, self.kappa3, self.l)
            x_vel = self.J @ np.append(u[0],[0,0])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
            
        elif self.stop == 7:
            pass      
        # Update the curvatures
        self.kappa1 += u[0] * dt 
        self.kappa2 += u[1] * dt
        self.kappa3 += u[2] * dt

        # TODO -> Solve the situation when kappas are zero in Homogenous matrix
        # Maybe when it is Zero try except and Raise an error
        self.kappa1 = np.clip(self.kappa1, self.kappa_min, self.kappa_max)
        self.kappa2 = np.clip(self.kappa2, self.kappa_min, self.kappa_max)
        self.kappa3 = np.clip(self.kappa3, self.kappa_min, self.kappa_max)

        # To check which curvature value are at the limits
        self.stop = 0
        k1 = self.kappa1 <= self.kappa_min or self.kappa1 >= self.kappa_max
        k2 = self.kappa2 <= self.kappa_min or self.kappa2 >= self.kappa_max
        k3 = self.kappa3 <= self.kappa_min or self.kappa3 >= self.kappa_max
        
        if k1:
            self.stop = 1           
        elif k2:
            self.stop = 2           
        elif k3:
            self.stop = 3
        if k1 and k2:
            self.stop = 4
        elif k1 and k3:
            self.stop = 5
        elif k2 and k3:
            self.stop = 6 
        if k1 and k2 and k3:
            self.stop = 7
        if self.observation_space.contains([new_x, new_y]):
            pass
        else:
            # Clip the states to avoid the robot to go out of the workspace
            self.overshoot0 += 1
            # print(new_x, new_y)
            new_x, new_y = self.observation_space.clip([new_x,new_y])
            # print(new_x, new_y)
            # TODO: When it is clipped, then write a algorithm to fill the empy trajectory between before clip and after clip

        if self.observation_space.contains([goal_x, goal_y]):
            new_goal_x, new_goal_y = goal_x, goal_y
        else:
            # Clip the states to avoid the robot to go out of the workspace
            self.overshoot1 += 1
            new_goal_x, new_goal_y = self.observation_space.clip([goal_x,goal_y])

        # States of the robot in numpy array
        self.state = np.array([new_x,new_y,new_goal_x,new_goal_y])
        
        return self._get_obs(), -1*self.costs, done, {} # Return the observation, the reward (-costs) and the done flag
    
    def step_minus_weighted_euclidean(self,u): # reward is -(e^2)
        
        x,y,goal_x,goal_y = self.state # Get the current state as x,y,goal_x,goal_y
        
        # global variables to be used in the reward function
        global new_x 
        global new_y
        global new_goal_x
        global new_goal_y

        dt =  self.dt # Time step
        
        u = np.clip(u, -self.kappa_dot_max, self.kappa_dot_max) # Clip the input to the range of the -1,1
        
        self.error = math.sqrt(((goal_x-x)**2)+((goal_y-y)**2)) # Calculate the error squared
        self.costs = 0.7 * self.error # Set the cost (reward) to the error squared
        
        # Just to show if the robot is moving along the goal or not
        if self.error < self.previous_error:
            #self.costs -= 1
            # UNCOMMENT HERE !!!!!!!
            pass
            # print("=========================POSITIVE MOVE=========================")
            
        
        self.previous_error = self.error # Update the previous error
        
        # if the error is less than 0.01, the robot is close to the goal and returns done
        if self.error <= 0.01:
            done = True
        else :
            done = False
         
        
        # This if and else statement is to avoid the robot to move if the kappas are at the limits
        if self.stop == 0:
            self.J = jacobian_matrix(self.delta_kappa, self.kappa1, self.kappa2, self.kappa3, self.l)
            x_vel = self.J @ u
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
            
        elif self.stop == 1:
            self.J = jacobian_matrix(self.delta_kappa, self.kappa1, self.kappa2, self.kappa3, self.l)
            x_vel = self.J @ np.append([0],u[1:3])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
        
        elif self.stop == 2:
            self.J = jacobian_matrix(self.delta_kappa, self.kappa1, self.kappa2, self.kappa3, self.l)
            x_vel = self.J @ np.append(np.append(u[0],[0]),u[2])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
            
        elif self.stop == 3:
            self.J = jacobian_matrix(self.delta_kappa, self.kappa1, self.kappa2, self.kappa3, self.l)
            x_vel = self.J @ np.append(u[0:2],[0])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
            
        elif self.stop == 4:
            self.J = jacobian_matrix(self.delta_kappa, self.kappa1, self.kappa2, self.kappa3, self.l)
            x_vel = self.J @ np.append([0,0],u[2])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
        
        elif self.stop == 5:
            self.J = jacobian_matrix(self.delta_kappa, self.kappa1, self.kappa2, self.kappa3, self.l)
            x_vel = self.J @  np.append(np.append([0],u[1]),[0])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
        
        elif self.stop == 6:
            self.J = jacobian_matrix(self.delta_kappa, self.kappa1, self.kappa2, self.kappa3, self.l)
            x_vel = self.J @ np.append(u[0],[0,0])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
            
        elif self.stop == 7:
            pass

        # Update the curvatures
        self.kappa1 += u[0] * dt 
        self.kappa2 += u[1] * dt
        self.kappa3 += u[2] * dt

        # TODO -> Solve the situation when kappas are zero in Homogenous matrix
        # Maybe when it is Zero try except and Raise an error
        self.kappa1 = np.clip(self.kappa1, self.kappa_min, self.kappa_max)
        self.kappa2 = np.clip(self.kappa2, self.kappa_min, self.kappa_max)
        self.kappa3 = np.clip(self.kappa3, self.kappa_min, self.kappa_max)

        # To check which curvature value are at the limits
        self.stop = 0
        k1 = self.kappa1 <= self.kappa_min or self.kappa1 >= self.kappa_max
        k2 = self.kappa2 <= self.kappa_min or self.kappa2 >= self.kappa_max
        k3 = self.kappa3 <= self.kappa_min or self.kappa3 >= self.kappa_max
        
        if k1:
            self.stop = 1
            
        elif k2:
            self.stop = 2
            
        elif k3:
            self.stop = 3
        
        if k1 and k2:
            self.stop = 4
        
        elif k1 and k3:
            self.stop = 5
        
        elif k2 and k3:
            self.stop = 6
            
        if k1 and k2 and k3:
            self.stop = 7
        
        if self.observation_space.contains([new_x, new_y]):
            pass
        else:
            # Clip the states to avoid the robot to go out of the workspace
            self.overshoot0 += 1
            # print(new_x, new_y)
            new_x, new_y = self.observation_space.clip([new_x,new_y])
            # print(new_x, new_y)
            # TODO: When it is clipped, then write a algorithm to fill the empy trajectory between before clip and after clip

        if self.observation_space.contains([goal_x, goal_y]):
            new_goal_x, new_goal_y = goal_x, goal_y
        else:
            # Clip the states to avoid the robot to go out of the workspace
            self.overshoot1 += 1
            new_goal_x, new_goal_y = self.observation_space.clip([goal_x,goal_y])     
        # States of the robot in numpy array
        self.state = np.array([new_x,new_y,new_goal_x,new_goal_y])
        
        return self._get_obs(), -1*self.costs, done, {} # Return the observation, the reward (-costs) and the done flag

    def step_distance_based(self,u): # reward is du-1 - du
        
        x,y,goal_x,goal_y = self.state # Get the current state as x,y,goal_x,goal_y
        global new_x 
        global new_y
        global new_goal_x
        global new_goal_y

        dt =  self.dt # Time step
        
        u = np.clip(u, -self.kappa_dot_max, self.kappa_dot_max) # Clip the input to the range of the -1,1
        
        self.error = math.sqrt(((goal_x-x)**2)+((goal_y-y)**2)) # Calculate the error squared
        
        # Just to show if the robot is moving along the goal or not
        if self.error < self.previous_error:
            #self.costs -= 1
            # UNCOMMENT HERE !!!!!!!
            pass
            # print("=========================POSITIVE MOVE=========================")
        
        if self.error == self.previous_error:
            self.costs = -100
        else:
            if self.error <= 0.025:
                self.costs = 200
            elif self.error <= 0.05:
                self.costs = 150
            elif self.error <= 0.1:
                self.costs = 100
            else:
                self.costs = 1000*(self.previous_error - self.error) # Set the cost (reward) du-1 - du
        
        self.previous_error = self.error
        
        # if the error is less than 0.01, the robot is close to the goal and returns done
        if self.error <= 0.010:
            done = True
        else :
            done = False
         
        
        # This if and else statement is to avoid the robot to move if the kappas are at the limits
        if self.stop == 0:
            self.J = jacobian_matrix(self.delta_kappa, self.kappa1, self.kappa2, self.kappa3, self.l)
            x_vel = self.J @ u
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
            
        elif self.stop == 1:
            self.J = jacobian_matrix(self.delta_kappa, self.kappa1, self.kappa2, self.kappa3, self.l)
            x_vel = self.J @ np.append([0],u[1:3])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
        
        elif self.stop == 2:
            self.J = jacobian_matrix(self.delta_kappa, self.kappa1, self.kappa2, self.kappa3, self.l)
            x_vel = self.J @ np.append(np.append(u[0],[0]),u[2])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
            
        elif self.stop == 3:
            self.J = jacobian_matrix(self.delta_kappa, self.kappa1, self.kappa2, self.kappa3, self.l)
            x_vel = self.J @ np.append(u[0:2],[0])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
            
        elif self.stop == 4:
            self.J = jacobian_matrix(self.delta_kappa, self.kappa1, self.kappa2, self.kappa3, self.l)
            x_vel = self.J @ np.append([0,0],u[2])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
        
        elif self.stop == 5:
            self.J = jacobian_matrix(self.delta_kappa, self.kappa1, self.kappa2, self.kappa3, self.l)
            x_vel = self.J @  np.append(np.append([0],u[1]),[0])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
        
        elif self.stop == 6:
            self.J = jacobian_matrix(self.delta_kappa, self.kappa1, self.kappa2, self.kappa3, self.l)
            x_vel = self.J @ np.append(u[0],[0,0])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
            
        elif self.stop == 7:
            pass

        # Update the curvatures
        self.kappa1 += u[0] * dt 
        self.kappa2 += u[1] * dt
        self.kappa3 += u[2] * dt

        # TODO -> Solve the situation when kappas are zero in Homogenous matrix
        # Maybe when it is Zero try except and Raise an error
        self.kappa1 = np.clip(self.kappa1, self.kappa_min, self.kappa_max)
        self.kappa2 = np.clip(self.kappa2, self.kappa_min, self.kappa_max)
        self.kappa3 = np.clip(self.kappa3, self.kappa_min, self.kappa_max)

        # To check which curvature value are at the limits
        self.stop = 0
        k1 = self.kappa1 <= self.kappa_min or self.kappa1 >= self.kappa_max
        k2 = self.kappa2 <= self.kappa_min or self.kappa2 >= self.kappa_max
        k3 = self.kappa3 <= self.kappa_min or self.kappa3 >= self.kappa_max
        
        if k1:
            self.stop = 1
            
        elif k2:
            self.stop = 2
            
        elif k3:
            self.stop = 3
        
        if k1 and k2:
            self.stop = 4
        
        elif k1 and k3:
            self.stop = 5
        
        elif k2 and k3:
            self.stop = 6
            
        if k1 and k2 and k3:
            self.stop = 7
        
        if self.observation_space.contains([new_x, new_y]):
            pass
        else:
            # Clip the states to avoid the robot to go out of the workspace
            self.overshoot0 += 1
            new_x, new_y = self.observation_space.clip([new_x,new_y])
        if self.observation_space.contains([goal_x, goal_y]):
            new_goal_x, new_goal_y = goal_x, goal_y
        else:
            # Clip the states to avoid the robot to go out of the workspace
            self.overshoot1 += 1
            new_goal_x, new_goal_y = self.observation_space.clip([goal_x,goal_y])

        
        # States of the robot in numpy array
        self.state = np.array([new_x,new_y,new_goal_x,new_goal_y])
        
        return self._get_obs(), self.costs, done, {} # Return the observation, the reward (-costs) and the done flag
   
    def reset(self):

       self.kappa1 = np.random.uniform(low=-4, high=16)
       self.kappa2 = np.random.uniform(low=-4, high=16)
       self.kappa3 = np.random.uniform(low=-4, high=16)
    
       T3_cc = three_section_robot(self.kappa1, self.kappa2, self.kappa3, self.l) # Generate the position of the tip of the robot
       x,y = np.array([T3_cc[0,3],T3_cc[1,3]]) # Extract the x and y coordinates of the tip

       self.target_k1 = np.random.uniform(low=-4, high=16) # 6.2 # np.random.uniform(low=-4, high=16)
       self.target_k2 = np.random.uniform(low=-4, high=16) # 6.2 # np.random.uniform(low=-4, high=16)
       self.target_k3 = np.random.uniform(low=-4, high=16) # 6.2 # np.random.uniform(low=-4, high=16)
       
       T3_target = three_section_robot(self.target_k1,self.target_k2,self.target_k3, self.l) # Generate the target point for the robot
       goal_x,goal_y = np.array([T3_target[0,3],T3_target[1,3]]) # Extract the x and y coordinates of the target
       
       self.state = x,y,goal_x,goal_y # Update the state of the robot
       
       self.last_u = None
       return self._get_obs()
    
    def _get_obs(self):
        x,y,goal_x,goal_y = self.state
        return np.array([x,y,goal_x,goal_y],dtype=np.float32)
    
    def render_calculate(self):
        # current state
        # section 1 calculation
        T1_cc = trans_mat_cc(self.kappa1,self.l[0])
        T1_tip = np.reshape(T1_cc[len(T1_cc)-1,:],(4,4),order='F');
        # section 2 calculation
        T2 = trans_mat_cc(self.kappa2,self.l[1]);
        T2_cc = couple_transformations(T2,T1_tip);
        T2_tip = np.reshape(T2_cc[len(T2_cc)-1,:],(4,4),order='F');
        # section 3 calculation
        T3 = trans_mat_cc(self.kappa3,self.l[2]);
        T3_cc = couple_transformations(T3,T2_tip);

        self.position_dic['Section1']['x'].append(T1_cc[:,12])
        self.position_dic['Section1']['y'].append(T1_cc[:,13])
        self.position_dic['Section2']['x'].append(T2_cc[:,12])
        self.position_dic['Section2']['y'].append(T2_cc[:,13])
        self.position_dic['Section3']['x'].append(T3_cc[:,12])
        self.position_dic['Section3']['y'].append(T3_cc[:,13])
        

    def render_init(self):
        # This function is used to plot the robot in the environment (both in start and end state)
        self.fig = plt.figure()
        self.fig.set_dpi(75);
        self.ax = plt.axes();


    def render_update(self,i):
        self.ax.cla()
        # Plot the trunk with three sections and point the section seperation
        self.ax.plot([-0.025, 0.025],[0,0],'black',linewidth=5)
        self.ax.plot(self.position_dic['Section1']['x'][i],self.position_dic['Section1']['y'][i],'b',linewidth=3)
        self.ax.plot(self.position_dic['Section2']['x'][i],self.position_dic['Section2']['y'][i],'r',linewidth=3)
        self.ax.plot(self.position_dic['Section3']['x'][i],self.position_dic['Section3']['y'][i],'g',linewidth=3)
        self.ax.scatter(self.position_dic['Section3']['x'][i][-1],self.position_dic['Section3']['y'][i][-1],linewidths=5,color = 'black')

        # Plot the target point and trajectory of the robot
        self.ax.scatter(self.state[2],self.state[3],100, marker= "x",linewidths=2, color = 'red')
        self.ax.set_title(f"The time elapsed in the simulation is {round(self.time,2)} seconds.")
        self.ax.set_xlabel("X - Position [m]")
        self.ax.set_ylabel("Y - Position [m]")
        self.ax.set_xlim([-0.4, 0.4])
        self.ax.set_ylim([-0.4, 0.4])

    
    def render(self):
        ani = FuncAnimation(fig = self.fig, func = self.render_update,frames=np.shape(self.position_dic['Section1']['x'])[0], interval = 20)
        # fig.suptitle('Helix Trajectory Animation', fontsize=14)
        return ani
        
        
    def visualization(self,x_pos,y_pos):
        # This function is used to plot the robot in the environment (both in start and end state)

        # Start state
        # section 1 calculation
        T1_cc = trans_mat_cc(self.start_kappa[0],self.l[0])
        T1_tip = np.reshape(T1_cc[len(T1_cc)-1,:],(4,4),order='F');
        # section 2 calculation
        T2 = trans_mat_cc(self.start_kappa[1],self.l[1]);
        T2_cc = couple_transformations(T2,T1_tip);
        T2_tip = np.reshape(T2_cc[len(T2_cc)-1,:],(4,4),order='F');
        # section 3 calculation
        T3 = trans_mat_cc(self.start_kappa[2],self.l[2]);
        T3_cc = couple_transformations(T3,T2_tip);

        # Plot the trunk with three sections and point the section seperation
        plt.plot([-0.025, 0.025],[0,0],'black',linewidth=5)
        plt.plot(T1_cc[:,12],T1_cc[:,13],'b',linewidth=3)
        #plt.scatter(T1_cc[-1,12],T1_cc[-1,13],linewidths=5,color = 'black')
        plt.plot(T2_cc[:,12],T2_cc[:,13],'r',linewidth=3)
        #plt.scatter(T2_cc[-1,12],T2_cc[-1,13],linewidths=5,color = 'black')
        plt.plot(T3_cc[:,12],T3_cc[:,13],'g',linewidth=3)
        plt.scatter(T3_cc[-1,12],T3_cc[-1,13],linewidths=5,color = 'orange',label='Initial Point')

        # End state
        # section 1 calculation
        T1_cc = trans_mat_cc(self.kappa1,self.l[0])
        T1_tip = np.reshape(T1_cc[len(T1_cc)-1,:],(4,4),order='F');
        # section 2 calculation
        T2 = trans_mat_cc(self.kappa2,self.l[1]);
        T2_cc = couple_transformations(T2,T1_tip);
        T2_tip = np.reshape(T2_cc[len(T2_cc)-1,:],(4,4),order='F');
        # section 3 calculation
        T3 = trans_mat_cc(self.kappa3,self.l[2]);
        T3_cc = couple_transformations(T3,T2_tip);

        # Plot the trunk with three sections and point the section seperation
        plt.plot(T1_cc[:,12],T1_cc[:,13],'b',linewidth=3)
        #plt.scatter(T1_cc[-1,12],T1_cc[-1,13],linewidths=5,color = 'black')
        plt.plot(T2_cc[:,12],T2_cc[:,13],'r',linewidth=3)
        #plt.scatter(T2_cc[-1,12],T2_cc[-1,13],linewidths=5,color = 'black')
        plt.plot(T3_cc[:,12],T3_cc[:,13],'g',linewidth=3)
        plt.scatter(T3_cc[-1,12],T3_cc[-1,13],linewidths=5,color = 'black')        
        
        # Plot the target point and trajectory of the robot
        plt.scatter(self.state[2],self.state[3],100, marker= "x",linewidths=4, color = 'red',label='Target Point')
        plt.scatter(x_pos,y_pos,25,linewidths=0.03,color = 'blue',alpha=0.2)
        plt.xlim([-0.4, 0.4])
        plt.ylim([-0.4, 0.4])
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=15)
        plt.grid(which='major',linewidth=0.7)
        plt.grid(which='minor',linewidth=0.5)
        # Show the minor ticks and grid.
        plt.minorticks_on()