
import numpy as np
import random



class ReplayBuffer(object):
    
    def __init__(self, size, env):
        #number of samples stored total at a time
        self.size = size 
        self.env = env
        #how many time steps to store a particular transition sample
        self.store_for_steps = 1
        #indices of the samples
        self.next_idx = 0
        self.num_in_buffer = 0
        #
        #storage of transition info
        self.obs_state = np.empty((self.size,self.env.state.shape[0],self.env.state.shape[1]))
        self.obs_action = np.empty((self.size),dtype=int)
        self.obs_reward = np.empty((self.size))
        #self.next_obs_state = np.empty((self.size,3))
        self.obs_done = np.empty((self.size),dtype=bool)
        
    #check whether the required batch size can be managed 
    def can_sample(self, batch_size):
        return batch_size + 1 <= self.num_in_buffer
    
    def sample_n_unique(self, sampling_func, batch_size):
        #returns an array of (batch_size) unique indices to be used for sampling
        idxes =[]
        while len(idxes) < batch_size:
            a = sampling_func() 
            if a not in idxes:
                idxes.append(a)
        #print('idxs',idxes)
        return idxes
    #
    #
    def sample(self, batch_size):
        #raise an error if not true
        assert self.can_sample(batch_size)
        
        idxes = self.sample_n_unique(
            lambda: random.randint(0, self.num_in_buffer - 2), batch_size)
        
        states = self.obs_state[idxes]
        actions = self.obs_action[idxes]
        next_states = [ self.obs_state[idx+1] for idx in idxes]
        rewards = self.obs_reward[idxes]
        done_vals = self.obs_done[idxes]
        
        return states, actions, rewards, next_states, done_vals
        
    def store(self, current_st, action, reward,  done, prefill):
        
        self.obs_state[self.next_idx] = current_st
        self.obs_action[self.next_idx] = action
        self.obs_reward[self.next_idx] = reward
        self.obs_done[self.next_idx] = done
        
        self.next_idx +=1 
    
        #implementing the preriodicity of a ring buffer 
        self.next_idx = self.next_idx % self.size
        
        #if not prefill:
         #   self.next_idx = self.next_idx + 100
        
        #update info on how filled is the buffer
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)
    
    def reset(self):
        self.next_idx = 0
        self.num_in_buffer = 0
        #
        #storage of transition info
        self.obs_state = np.empty((self.size,self.env.state.shape[0],self.env.state.shape[1]))
        self.obs_action = np.empty((self.size),dtype=int)
        self.obs_reward = np.empty((self.size))
        #self.next_obs_state = np.empty((self.size,3))
        self.obs_done = np.empty((self.size),dtype=bool)
        
        
        
        
        
        
        