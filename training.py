import random

import numpy as np
import pandas as pd

import tensorflow as tf

from tqdm import tqdm
from copy import deepcopy

from multiprocessing.pool import ThreadPool
from multiprocessing import Pool

from model import DeepQNetwork
from env import Environment
from replay_buffer import InMemoryReplayBuffer

def sample_random_episode(env):
    #we reset the environment and store the initial state observation
    env.reset()
    previous_state = env.observation
    episode_batch = []

    #while the episode has not reached a terminal state
    while not env.terminal:
        action, successor, reward, terminal = env.do_random_action()

        #we store the s,a,r,s_prime,t element
        episode_batch += [(previous_state, action, reward, successor, terminal)]

        #we overwrite the previous state
        previous_state = successor

    return episode_batch

def random_initial_sampling(buffer, env, params):
    with tf.device('/cpu:0'):
        with tqdm(total=params['buffer_size_in_batches']*params['batch_size'],desc='Filling initial buffer') as pbar:
            #we create environments to sample with synchronously with thread pooling
            envs = [deepcopy(env) for _ in range(5)] + [env]

            with Pool(6) as pool:
                while len(buffer) < params['buffer_size_in_batches']*params['batch_size']:
                    episode_batches = pool.starmap(sample_random_episode,[(env,) for env in envs])
                    episode_batches = [elem for stack in episode_batches for elem in stack]

                    #reshape the data into a list of tensors with the batch
                    episode_batches = [tf.stack([episode[i] for episode in episode_batches]) for i in range(5)]

                    pbar.update(int(episode_batches[0].shape[0]))

                    #add the sampled episode to the buffer
                    buffer.add(episode_batches)
            
            del envs

def training_loop(model, target_model, buffer, params, epoch):
    with tf.device('/gpu:0'):
        #we loop over the number of training steps we wish to do on the current buffer per epoch
        td_errors = []

        #we sample batches from the buffer
        data = buffer.sample(batch_count=params['batches_per_epoch'], batch_size=params['batch_size'])

        for batch in tqdm(data, desc=f'Training epoch {epoch}'):
            states, actions, rewards, successors, terminals = batch

            with tf.GradientTape() as tape:
                q_vals = model(states)
                q_val_primes = target_model(successors)

                #we calculate the td error for the current episode
                for q_val, q_val_prime, action, reward, terminal in zip(q_vals, q_val_primes, actions, rewards, terminals):
                    if bool(terminal):
                        td_errors += [tf.keras.losses.mean_squared_error(reward, tf.reduce_max(q_val[int(action)]))]

                    else:
                        td_errors += [tf.keras.losses.mean_squared_error(reward + params['gamma'] * tf.reduce_max(q_val_primes[int(action)]), tf.reduce_max(q_val[int(action)]))]

                #we train averaging over the entire sample from buffer
                gradient = tape.gradient(td_errors, model.trainable_variables)

                model.optimizer.apply_gradients(zip(gradient, model.trainable_variables))
                
        #we ensure the deletion of the sample just to make sure
        del data

        return float(tf.reduce_mean(td_errors))

def target_update(model, target_model, params):
    for target_variable, model_variable in zip(target_model.trainable_variables, model.trainable_variables):
        target_variable.assign(params['polyak_avg_fac'] * target_variable + (1 - params['polyak_avg_fac']) * model_variable)

def sample_episode_step(env, model, epsilon):
    #we reset the environment and store the initial state observation
    env.reset()
    episode_batch = []
    previous_state = env.observation

    #while the episode has not reached a terminal state
    while not env.terminal:
        policy = model(tf.expand_dims(previous_state, axis=0)) #we retrieve the policy

        #we grab the index of the policy with the highest prob
        action, successor, reward, terminal = tf.constant([int(np.argmax(tf.squeeze(policy).numpy()))],dtype=tf.int32), None, None, None

        #if random <= episilon value at current epoch then we proceed with a random action
        if random.random() <= epsilon:
            action, successor, reward, terminal = env.do_random_action()

        else:
            successor, reward, terminal = env.step(int(np.argmax(tf.squeeze(policy).numpy())))

        #we store the s,a,r,s_prime,t element
        episode_batch += [(previous_state, action, reward, successor, terminal)]

        #we overwrite the previous state
        previous_state = successor

    return episode_batch

def sampling(env, model, buffer, params, epoch):
    '''
    ADD
    '''
    
    #we calculate the epsilon for the current epoch here
    epsilon = params['epsilon']*params['epsilon_decay']**epoch
    
    with tf.device('/cpu:0'):
        with tqdm(total=int(params['buffer_size_in_batches']*params['batch_size']*params['replay_ratio']),desc='Filling buffer with new elements') as pbar:
            count = 0
            envs = [deepcopy(env) for _ in range(5)] + [env]

            #we loop while sampling until we have updated "replay_ratio" fraction of the replay buffer with new samples
            with ThreadPool(6) as pool:
                while count < params['replay_ratio'] * params['buffer_size_in_batches'] * params['batch_size']:
                    episode_batches = pool.starmap(sample_episode_step,[(env, model, epsilon) for env in envs])
                    episode_batches = [elem for stack in episode_batches for elem in stack]

                    #reshape the data into a list of tensors with the batch
                    episode_batches = [tf.stack([episode[i] for episode in episode_batches]) for i in range(5)]

                    #we update the counter(s)
                    count += int(episode_batches[0].shape[0])
                    pbar.update(int(episode_batches[0].shape[0]))

                    #add the sampled episode to the buffer
                    buffer.add(episode_batches)
            
            del envs

def train(params : dict, path : str, model_name : str, model_id : int, df):
    '''
    ADD

    :param params (dict): A dictionary of hyperparameters and static parameters for this model type (cf. README and bay_hyperparam_search.py).
    :param path (str): The path to the directory in which to store the weights, optimizer, and training data.
    :param model_affix (str): The affix for the model name and id, ideally of form "<name>_id<id>", the epoch number will be added by this function.
    :param record (bool): A bool whether to save the model weights, optimizer, and training data for True, if set to false the progress is not stored but used for search.

    :returns (tuple): Returns a tuple of data in form of (avg_td_errors : list, traintimes : list), each list is a list of floats.
    '''

    env = Environment(rgb=params['rgb'])

    model = DeepQNetwork(
        tf.keras.optimizers.SGD(learning_rate=params['learning_rate']),
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

    target_model = DeepQNetwork(
        tf.keras.optimizers.SGD(learning_rate=params['learning_rate']),
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
    random_initial_sampling(buffer, env, params)

    #we declare lists to store td_error
    avg_td_errors = []

    #we loop over the total number of epochs we wish to train for
    for epoch in range(params['epochs']):
        if epoch > 0:
            #we fill the buffer with new samples for the next epoch using the current network
            sampling(env, model, buffer, params, epoch)
            
        avg_td_error = training_loop(model, target_model, buffer, params, epoch+1)

        #after the epoch we update the target network with polyak-averaging
        target_update(model, target_model, params)

        #we save the model and optimizer state after each epoch
        model.save(path, path_affix=f'{model_name}_id{model_id}_ep{epoch+1}_')

        #we store training data in df
        df = pd.concat([df,pd.DataFrame([[model_name, model_id, epoch+1, avg_td_error]], columns=df.columns)])
        df.to_csv(path+'data.csv', index=None)

        print(f'Finished epoch {epoch+1}/{params["epochs"]} with an average td error of {avg_td_error}.')
    
    #we ensure proper garbage collection when training is done and data was saved
    del buffer
    del model
    del df
    del env