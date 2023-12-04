import gym

class Environment:
    '''Wrapper class for the Lunar Lander environment from gym.'''
    
    def __init__(self, visual=False):
        '''
        :param visual (bool): A parameter whether the observation created by the environment should be visual or a (8,) tensor
        '''
        
        self.env = lunar_lander_env = gym.make("LunarLander-v2", continuous=False, gravity=-10.0, enable_wind=False, wind_power=15.0)
        
    def step(self):
        pass
    
    def observation(self):
        pass
    
    def reset(self):
        pass