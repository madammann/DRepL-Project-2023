import numpy as np
import tensorflow as tf

import gym

from PIL import Image

class Environment:
    '''Wrapper class for the Lunar Lander environment from gym.'''
    
    def __init__(self, visual=True, rgb=True):
        '''
        Constructor method for the environment, defaults to creating an environment with image observations in greyscale of shape (50,75,1).
        
        :param visual (bool): A Boolean whether the observation created by the environment should be visual (50,75,rgb) for True, or a (8,) tensor.
        :param rgb (bool): A Boolean whether to turn the image observation rgb or not.
        '''
        
        self.visual = visual
        self.rgb = rgb
        
        self.env = gym.make("LunarLander-v2", continuous=False, gravity=-10.0, enable_wind=False, wind_power=15.0, render_mode='rgb_array')
        self.env.reset()
        self.terminal = False
        
    def step(self, action : int):
        '''
        Method for taking a step action in the environment.
        
        :returns (tuple): (observation : tf.tensor, reward : tf.tensor, terminal : tf.tensor) tuple with observation and reward in float32 dtype and terminal in bool dtype.
        '''
        
        observation, reward, terminal, truncated, info = self.env.step(action)
        self.terminal = terminal
        
        if self.visual:
            return self._get_image(), tf.constant([reward], dtype=tf.float32), tf.constant([terminal], dtype=tf.bool)
        
        #otherwise we return the observation from the environment that is not an image
        return tf.constant([observation], dtype=tf.float32), tf.constant([reward], dtype=tf.float32), tf.constant([terminal], dtype=tf.bool)
    
    def reset(self):
        '''
        Reset wrapper for the environment.
        '''
        
        self.env.reset()
        self.terminal = False
        
    def do_random_action(self):
        '''
        Method for performing a random action in the environment.
        
        :returns (tuple): (action : tf.tensor, observation : tf.tensor, reward : tf.tensor, terminal : tf.tensor) tuple with observation, action, and reward in float32 dtype and terminal in bool dtype.
        '''
        
        action = self.env.action_space.sample()
        
        return tf.constant([action], dtype=tf.float32), *self.step(action)
    
    @property
    def observation(self):
        return self._get_image()
    
    def close(self):
        '''
        Close wrapper for the environment.
        '''
        
        self.env.close()
        
    def _get_image(self):
        '''
        Method for generating the downscaled and potentially greyscaled image tensor for the model.
        
        :returns (tf.tensor): A tensor of shape (50,75,1) for greyscale or (50,75,3) for RGB.
        '''
        
        img = self.env.render()
        
        #we rescale the image down from (400,600) to (50,75) and make it greyscale
        img = Image.fromarray(img).resize((img.shape[1] // 8, img.shape[0] // 8))
        img = img.convert(mode='L') if not self.rgb else img
        
        #we return the image as tf.tensor of shape (50,75,1) for greyscale or (50,75,3) for non greyscale RGB
        return tf.constant(np.expand_dims(np.array(img), -1), dtype=tf.float32) if not self.rgb else tf.constant(np.array(img), dtype=tf.float32)
    
def visualize_episodes(example_episodes : list):
    '''
    Function for visualizing and rendering inside pygame some example episodes selected.
    
    :param example_episodes (list): A list of lists of actions as integers representing the action taken in each episode.
    '''

    env = gym.make("LunarLander-v2", continuous=False, gravity=-10.0, enable_wind=False, wind_power=15.0, render_mode='human')
    
    #iterate over a list of example episodes (list of list of actions) and render them out 
    for actions_done in example_episodes:
        env.reset()
        
        #loop to show a specific episode
        for action in actions_done:
            env.render()
            try:
                env.step(action)
                
            #if weird non-deterministic behavior occurs, we break the episode loop
            except:
                break