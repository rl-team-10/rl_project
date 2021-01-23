import numpy as np
from numpy import newaxis
import math
import random
import gym # is this allowed
from gym import spaces
from gym.utils import seeding
from gym.spaces import Dict, Discrete, Box, Tuple # is this allowed?
from EoM import EoM

class Environ3D():
    # Main Environment Class
    # TO DO:
    # implement continous action space (maybe use gym?)
    # implement rotation of elipsoid (via inertial forces)
    # implement sensible update rule for velocity


    # very simple, discrete example for now (delibarately 2d)
    action_Space = np.array([[40.0,0.0,0.0],[0.0,40.0,0.0],[0.0,0.0,40.0],[-40.0,0.0,0.0],[0.0,-40.0,0.0],[0.0,0.0,-40.0],[0.0,0.0,0.0]])
    def __init__(self,seed):
        #initialize agent
         self.seed(seed)
         self.done = False
        #initialize asteroid
         self.ellipsoid_Center = np.array([0.0,0.0,0.0])
        #self.ellipsoid_Radius = 1`
         self.ellipsoid_Semiaxis_a = np.random.uniform(100,2000,1)
         self.ellipsoid_Semiaxis_b = 1.1*self.ellipsoid_Semiaxis_a 
         self.ellipsoid_Semiaxis_c = 3.0*self.ellipsoid_Semiaxis_a 
        #rn target point is above north pole of the sphere
         self.target_Point = self.ellipsoid_Center+ np.array([0.0,self.ellipsoid_Semiaxis_b+self.ellipsoid_Semiaxis_b/2.0,0.0])
        #rotation and ang vel are not used rn 
         self.ellipsoid_Axis_Of_Rotation = np.array([1.0,0.0,0.0])
         self.ellipsoid_Density = np.random.uniform(1500,3000,1)
         self.newton_Constant = 6.7*10**(-11)
         self.ellipsoid_Angular_Velocity_max = math.sqrt(4/3*self.newton_Constant*np.pi*self.ellipsoid_Density*self.ellipsoid_Semiaxis_b*self.ellipsoid_Semiaxis_c/self.ellipsoid_Semiaxis_a**2)
         self.ellipsoid_Angular_Velocity = np.random.uniform(self.ellipsoid_Angular_Velocity_max/4,self.ellipsoid_Angular_Velocity_max,3)
        #self.ellipsoid_Mass = self.mass_Ellipsoid(self.ellipsoid_Semiaxis_a, self.ellipsoid_Semiaxis_b, self.ellipsoid_Semiaxis_c)
         self.ellipsoid_Mass = self.ellipsoid_Density*np.pi*self.ellipsoid_Semiaxis_a*self.ellipsoid_Semiaxis_b*self.ellipsoid_Semiaxis_c*4.0/3.0
        #self.seed()
        #print(self.ellipsoid_Angular_Velocity_max)
         self.state = np.zeros((2,3))

    def seed(self,seed):
        #seed = self.seed
        np.random.seed(seed)
        #return [seed]

    #def mass_Ellipsoid(Semiaxis_a,Semiaxis_b,Semiaxis_c):
     #   mass = ellipsoid_Density*np.pi*Semiaxis_a*Semiaxis_b*Semiaxis_c*4.0/3.0
     #   return mass
    


    def step(self, action):
         action_Taken = self.action_Space[action].copy()
         position, velocity = self.state
        #print(self.state)
        #update position
        #why not first update the velocity?
         
        #update velocity (it's rudimentory now)
        #added simple gravity simulation
         distance_To_Center = np.linalg.norm(position-self.ellipsoid_Center)
         gravity_Accel = -1*self.ellipsoid_Mass*self.newton_Constant*(1/distance_To_Center**2)*position
         velocity += EoM(position,velocity,gravity_Accel,self.ellipsoid_Angular_Velocity,action_Taken) 
         position += velocity
         
        #collision (for done-ness)
         if ((position[0]**2)/(self.ellipsoid_Semiaxis_a**2)
                 + (position[1]**2)/(self.ellipsoid_Semiaxis_b**2)
                 + (position[2]**2)/(self.ellipsoid_Semiaxis_c**2)) < 1.0:
            self.done = True
            reward = -10
            #print("INSIDE")
         if np.linalg.norm(position-self.target_Point) < 0.1:
            reward = 0
         else:
            reward = - np.linalg.norm(position-self.target_Point) /100000
                #print(distance_To_Center)
        #print(position)

        #wrap up the step
         self.state = (position,velocity)
         return np.array(self.state), reward, self.done


    def reset(self):
        #should be uniformly sampled from the ellipsoid?? 
        # self.position = np.array([random.random()+1,random.random()+1,random.random()+1]
         self.position = np.random.uniform(-100000,100000,3)
        #should be random in sensible intervals
         self.velocity = np.random.uniform(-0.3,0.3,3)

         self.state = np.array([self.position, self.velocity])
         return np.array(self.state)
