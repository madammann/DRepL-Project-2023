import numpy as np
import pandas as pd

import tensorflow as tf

from datetime import datetime, timedelta
from tqdm import tqdm

from model import DeepQNetwork
from environment import Environment
from replay_buffer import InMemoryReplayBuffer

def train(params : dict, path : str, model_affix : str, record=True):
    '''
    ADD
    
    :param params (dict): A dictionary of hyperparameters and static parameters for this model type (cf. README and bay_hyperparam_search.py).
    :param path (str): The path to the directory in which to store the weights, optimizer, and training data.
    :param model_affix (str): The affix for the model name and id, ideally of form "<name>_id<id>", the epoch number will be added by this function.
    :param record (bool): A bool whether to save the model weights, optimizer, and training data for True, if set to false the progress is not stored but used for search.
    
    :returns (tuple): Returns a tuple of data in form of (avg_td_errors : list, traintimes : list), each list is a list of floats.
    '''
    
    optimizer = tf.keras.optimizers.SGD(learning_rate=params['learning_rate'])
    
    env = Environment(visual=params['visual'], rgb=params['rgb'])
    
    model = DeepQNetwork(
        optimizer,
        visual=params['visual'],
        rgb=params['rgb'],
        cnn_depth=params['cnn_depth'],
        mlp_layers=params['mlp_layers'],
        head_layers=params['head_layers'],
        filters=params['filters'],
        kernel_size=params['kernel_size'],
        k_init=params['k_init'],
        b_init=params['b_init']
    )
    
    buffer = InMemoryReplayBuffer(state_shape=(50, 75, 3) if params['rgb'] else (50, 75, 1), buffer_size=params['buffer_size_in_batches']*params['batch_size'])
    
    #we fill the initial replay buffer with random actions
    while len(buffer) < param['buffer_size']:
        episode_batch = []
        
        #we reset the environment and store the initial state observation
        env.reset()
        previous_state = env.observation
        
        #while the episode has not reached a terminal state
        while not env.terminal:
            action, sucessor, reward, terminal = env.do_random_action()
            
            #we store the s,a,r,s_prime,t element
            episode_batch += [(previous_state, action, reward, sucessor, terminal)]
            
            #we overwrite the previous state
            previous_state = sucessor
        
        #reshape the data into a list of tensors with the batch
        episode_batch = [tf.stack([episode[i] for episode in episode_batch]) for i in range(4)]
        
        #add the sampled episode to the buffer
        buffer.add(episode_batch)
    
    #we declare lists to store td_error and training times
    avg_td_errors = []
    training_times = []
    
    #we loop over the total number of epochs we wish to train for
    for epoch in param['epochs']:
        start_time = datetime.now()
        #we loop over the number of training steps we wish to do on the current buffer per epoch
        td_errors = []
        for batch in tqdm(param['batches_per_epoch'], desc=f'Training epoch {epoch}:'):
#             states, actions, rewards, successors, terminals = batch

#             with tf.GradientTape() as tape:
#             pred_values, pred_policies = self.model(states,training=True)
#             tar_pred_values, tar_policies = self.target_model(successors)

#        #we calculate the td error for the current episode (MISSING ACTION USAGE AND RIGHT FORMULA)
        # if bool(terminal):
            # td_errors += tf.keras.losses.meansquareerror(reward, model(state))
        # else:
            # td_errors += tf.keras.losses.meansquareerror(reward + params['gamma'] * model(successor), model(state))

#             gradient = tape.gradient(td_errors[-1], model.trainable_variables)

#             model.optimizer.apply_gradients(zip(gradient, model.trainable_variables))
        
        # avg_td_errors += [float(np.mean(td_errors))]
        
        #we fill the buffer with new samples for the next epoch using the current network
        count = 0
        #we loop while sampling until we have updated "replay_ratio" fraction of the replay buffer with new samples
        while count < params['replay_ratio'] * params['buffer_size_in_batches'] * params['batch_size']:
            pass
        
        training_times += [(datetime.now() - start).seconds]
        
    return avg_td_errors, training_times