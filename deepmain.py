import jax.numpy as jnp
import numpy as np
from buffer import ReplayBuffer
from environ import Environ3D
from netjit import DoubleDQN
from jax import jit
import matplotlib.pyplot as plt
import itertools

seed = 3
buffer_size = 3000

batch_size = 50
training_freq = 1
target_update_freq = 200
gamma = 0.04
explor_period = 10000

env = Environ3D(seed)
buffer = ReplayBuffer(buffer_size,env)
dqn = DoubleDQN(len(env.action_Space), buffer, buffer_size, batch_size, training_freq,\
               target_update_freq, gamma, explor_period, seed, env)
    
env.reset()

#######
#prefill buffer

prefill_buffer_size = 50000
buffer.reset()

for _ in range(prefill_buffer_size):
    
    action = np.random.randint(0,len(env.action_Space))
    current_state = np.copy(env.state)
    next_state, reward, done = env.step(action)
    buffer.store(current_state, action, reward, done, prefill = True)
    
    if done:
        env.reset()
    
#reset when prefilling is done
env.reset()

###########

#train the network
dqn.training = True

training_steps = 10000
total_reward = 0

x=[]
y=[]
z=[]

rewards = []
mean_rewards = []
episode_num = 0

i=0
debug = False
for step in range(training_steps):
     i+=1
     if i % 100 == 0:
         print("training episode no.", i)
         debug = True
     current_state = np.copy(env.state)
     action = dqn.choose_action(current_state, debug)
     next_state, reward, done = env.step(action)
     buffer.store(current_state, action, reward, done,  prefill = False)
     rewards.append(reward)
     debug = False
     #print(reward)
     if done or i % 150 == 0:
         mean_rewards.append(np.mean(rewards))
         episode_num += 1
         env.reset()
     
####### predict

dqn.training = True
     
env.reset()
print("startimng state:" , env.state)

i=0
debug = False
rewards_nav= []
mean_rewards_nav = []
states = []
done_list = []
while i < 10000:
     
     if i % 100 == 0:
         print("navigating...step no.", i)
         debug = True
     current_state = np.copy(env.state)
     action = dqn.choose_action(current_state, debug)
     next_state, reward, done = env.step(action)
     buffer.store(current_state, action, reward, done, prefill = False)
     total_reward += reward
     rewards_nav.append(reward)
     states.append(current_state)
     done_list.append(done)
     #print(dqn.step)
     debug = False
     x.append(next_state[0][0])
     y.append(next_state[0][1])
     z.append(next_state[0][2])
     
     i+=1
     
     if reward < -3:
         break
     
     if done: 
         break
     
        
print("finishing state:" , env.state)     
print('mean reward:',total_reward/i,'/n','steps taken:', i )
print('target:',env.target_Point)

filex = open('x_vals.txt','w')
filey = open('y_vals.txt','w')
filez = open('z_vals.txt','w')

for i in range(len(x)):
    filex.write(str(x[i])+'\n')
    filey.write(str(y[i])+'\n')
    filez.write(str(z[i])+'\n')
    
filex.close()
filey.close()
filez.close()
     
 ###########################################################################
 #mathplotlib shit (plots 3D now!)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.grid(True)
ax.plot(x,y,z, marker='.', markevery=5)
ax.plot(env.target_Point[0],env.target_Point[1], '-xr')
#trying to plot elipsoid
 #coefs = (environment.ellipsoid_Semiaxis_a, environment.ellipsoid_Semiaxis_b,
 #        environment.ellipsoid_Semiaxis_c)
coefs = (1/env.ellipsoid_Semiaxis_a**2,
     1/env.ellipsoid_Semiaxis_b**2,
     1/env.ellipsoid_Semiaxis_c**2)  # Coefficients in a0/c x**2 + a1/c y**2 + a2/c z**2 = 1
#Radii corresponding to the coefficients:
rx, ry, rz = 1/np.sqrt(coefs)
#Set of all spherical angles:
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
# Cartesian coordinates that correspond to the spherical angles:
# (this is the equation of an ellipsoid):
xe = rx * np.outer(np.cos(u), np.sin(v))
ye = ry * np.outer(np.sin(u), np.sin(v))
ze = rz * np.outer(np.ones_like(u), np.cos(v))
 # Plot:
ax.plot_surface(xe, ye, ze,  rstride=2, cstride=4, color='grey')
plt.show()
###########################################################################


episodes = np.linspace(0, episode_num,episode_num)

plt.plot(episodes, mean_rewards)
plt.xlabel('Number of episodes')
plt.ylabel('Mean reward received during episode')
plt.show()


'''
circle1 = plt.Circle((0, 0), 1.0, color='r')
fig = plt.figure()
   # ax = fig.gca(projection='3d')
ax = fig.gca()
ax.grid(True)
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
    
ax.plot(x,y ,'xb-')
ax.add_artist(circle1)
ax.plot(env.target_Point[0],env.target_Point[1], '-xr')
    #plt.axis('equal')
plt.show()
     
   '''  
     
     
     
    
    
