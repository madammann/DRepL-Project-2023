import gym

class Environment:
    '''Wrapper class for the Lunar Lander environment from gym.'''
    
    def __init__(self, visual=False, render=False):
        '''
        :param visual (bool): A parameter whether the observation created by the environment should be visual or a (8,) tensor
        '''
        
        render_mode = 'none'
        if render == True:
            render_mode = 'human'
        
        elif visual == True:
            render_mode = 'rgb_image'
            
        self.env = lunar_lander_env = gym.make("LunarLander-v2", continuous=False, gravity=-10.0, enable_wind=False, wind_power=15.0, render_mode=render_mode)
        
    def step(self, action):
        '''
        ADD
        '''
        
        observation, reward, terminal, truncated, info = env.step(action)
        
        if visual and not render:
            observation = self.env.render() #we obtain the rgb image for the CNN as observation
            
        return observation, reward, terminal #we omit truncated and info as they are unecessary for our purposes
    
    def reset(self):
        self.env.reset()
        
    def do_random_action(self):
        '''
        ADD
        '''
        action = self.env.action_space.sample()
        
        return self.step(action)
    
    def close(self):
        self.env.close() 
    
def visualize_episode(env, example_episodes : list):
    '''
    ADD
    '''

    #iterate over a list of example episodes (list of list of actions) and render them out 
    for actions_done in example_episodes:
        env.reset()
        terminal = False
        
        for action in actions_done:
            env.render()
            action = env.action_space.sample()
            observation, reward, terminal, truncated, info = env.step(action)