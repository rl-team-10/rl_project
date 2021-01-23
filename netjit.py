import jax.numpy as jnp 
from jax import random, lax
from jax.experimental import stax 
from jax.experimental.stax import Dense, Relu, FanOut, elementwise
from jax.experimental import optimizers
from buffer import ReplayBuffer
import numpy as np
from jax import grad, tree_util
import itertools
from jax import jit
from functools import partial

class LinSched(object):
    
    def __init__(self, timesteps):
        self.timesteps = timesteps
    
    def value(self, t):
        if t < self.timesteps / 2:
            value = self.value_calc(t, final = 0.3, init = 1.0)
        elif t < self.timesteps:
            value = self.value_calc(t, final = 0.01, init = 0.3)
        else:
            value = 0.08
        return value
    
    def value_calc(self, t, final, init):
        if t < self.timesteps:
            frac = min(t/(self.timesteps), 1.0)
            value = init + frac*(final - init)
            return value
'''    
def LRSched():
    init = 0.09
    decay = 0.09
    return lambda x: init*jnp.exp(-decay*x)
    '''

class DoubleDQN(object):
    
    def __init__ (self, num_actions, buffer, buffer_size, batch_size, training_freq, target_update_freq, \
                  gamma, explor_period, seed, env):
        self.num_actions = num_actions
        #training frequency for base q net
        self.training_freq = training_freq
        #update target q net every _ steps
        self.target_update_freq = target_update_freq
        #discount factor
        self.gamma = gamma
        self.buffer = buffer
        self.batch_size = batch_size
        #should we include addictional exploration prior to tranining ?
        self.explor_period = explor_period
        self.exploration = LinSched(self.explor_period)
        self.env = env
        
        self.seed = seed
        key = random.PRNGKey(self.seed)
        
        self.input_shape = (-1,) + (self.batch_size,) + (jnp.reshape(self.env.state, -1)).shape 
        self.norm = 1/len(jnp.reshape(self.env.state, -1))
        
        
        self.dueling = True
        
        #neural nets, same layer structure
        self.init, self._apply = self.q_network()
        #divide key to produce two distinct q nets
        key1, key2 = random.split(key)
        _, self.base_model = self.init(key1, self.input_shape)
        _, self.target_model = self.init(key2, self.input_shape) 
        self.apply=jit(self._apply)
        self.apply = self._apply
        #to use for base model
        
        decay_lr = 0.1
        init_lr = 0.0007
        
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(lambda x: init_lr*jnp.exp(-decay_lr*x),
                                                                          b1 = 0.9, b2 = 0.999,
                                                                          eps = 1e-5)
        #
        self.state = self.opt_init(self.base_model)
        
        self.step = itertools.count(0,1)
        self.training = True

    @partial(jit, static_argnums=(0,2))  
    def choose_action(self, obs_state, debug):
            step = next(self.step)
            if np.random.rand() < self.exploration.value(step):# and step < self.explor_period:
                action = np.random.randint(self.num_actions)
                if debug :
                    print("\n","action chosen randomly:", action)
            else: 
                if self.training:
                    self.update_base(step)
                if self.dueling:
                    value, adv = self.apply(self.base_model, jnp.reshape(obs_state, -1))
                    preds  = value + adv - jnp.mean(adv)
                    if debug:
                        print("\n","value; advatnage:", value, adv)
                else:
                    preds = self.apply(self.base_model,   (jnp.reshape(obs_state,-1)))
                action = np.argmax(preds)
                if step % self.target_update_freq == 0 and self.training:
                    self.update_target()
                if debug:
                    print( "action chosen by net: qvals =", preds, ", action chosen:", action)
            return action 
    
    def q_network(self):
        #no regression !
        if self.dueling:
            init, apply = stax.serial(
                            elementwise(lambda x: x/10000.0),
                            stax.serial(Dense(128), Relu, Dense(64), Relu ), #base layers
                            FanOut(2),
                            stax.parallel(
                                            stax.serial(Dense(32), Relu, Dense(1)), #state value
                                            stax.serial(Dense(32), Relu, Dense(self.num_actions))
                                           ) #advantage func
                            )
                            
        else:        
            init, apply = stax.serial(elementwise(lambda x: x/10000.0), Dense(64), Relu, \
                                      Dense(32), Relu, Dense(self.num_actions))
        
        return init, apply
    
    
    @partial(jit, static_argnums=(0,))   
    def update_base(self,step):
        states, actions, rewards, next_states, done_list = self.sample_batch()
       
        states = np.array([ jnp.reshape(state, -1) for state in states])
        next_states = np.array([ jnp.reshape(state, -1) for state in next_states])
        
        if self.dueling:
            value, adv = self.apply(self.target_model, next_states)
            next_q_vals  = value + adv - jnp.mean(adv)
        else:
            next_q_vals = self.apply(self.target_model, next_states)
            
            
        q_max = jnp.max(next_q_vals, axis = 1) 
        #update target 
        targets = rewards + self.gamma * q_max *( 1 - done_list) ##..this
        #print(rewards)
        targets = jnp.clip(targets, -2.0, 2.0)
        
        #assume independence of the target of the net params
        targets = lax.stop_gradient(targets)
        
        params = self.get_params(self.state) #calculating those explicitly in order to 
                                                    #differentiate w respect to them
        grads = grad(self.Bellman_loss)(params, states, actions, targets)
        
        grads = tree_util.tree_map(lambda f: jnp.clip(f, -10.0, 10.0), grads)
        #print(grads)
        
        self.state = self.opt_update(step, grads, self.state)
        #update params
        self.base_model = self.get_params(self.state)
        
        

    def sample_batch(self):
        return self.buffer.sample(self.batch_size)
        
    def update_target(self):
        #polayk averaging?
        self.target_model = self.get_params(self.state)
        
        
        
        
        #loss function that has grad directly applied to is has to
        #contain apply func
    @partial(jit, static_argnums=(0,))
    def Bellman_loss(self, params, states, actions, targets):
        if self.dueling:
            value, adv = self.apply(params, states)
            q_vals = value + adv - jnp.mean(adv) ##fit this to...
        else:
            q_vals = self.apply(params, states)
        #fitting the q vals of actions which have actually been taken by isolating these qvals
        q_vals_reshape = jnp.take_along_axis(q_vals, jnp.expand_dims(actions, axis=1), axis=1)
        
        return jnp.mean(self.loss(q_vals_reshape - targets))
    
    
    def loss(self, x):
        d = 1.0
        if jnp.all(jnp.abs(x) <= d) :
            return 0.5 * x**2 
        else:
            return (0.5*d**2 + d * (jnp.abs(x) - d))

        
        
        
