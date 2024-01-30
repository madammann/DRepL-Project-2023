import gc

import numpy as np
import tensorflow as tf

class InMemoryReplayBuffer:
    def __init__(self, state_shape=(50, 75, 3), action_shape=(1,), reward_shape=(1,), buffer_size=1000):
        '''
        Initialization method for the replay buffer.

        :param state_shape (tuple): A tuple representing the shape of the tensor for a state.
        :param action_shape (tuple): A tuple representing the shape of the tensor for an action.
        :param reward_shape (tuple): A tuple representing the shape of the tensor for a reward.
        :param buffer_size (int): The maximum size of the replay buffer, default is 100.000 elements.

        NOTE: buffer_size has to be compatible with available memory limits.
        '''

        #maximum size of the buffer
        self.buffer_size = buffer_size

        #shapes for each element in the sarst buffer, with state_shape being used for both state and successor state
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.reward_shape = reward_shape

        #initial empty tensors for each element
        self.states = np.zeros((0, *state_shape), dtype=np.int16)
        self.actions = np.zeros((0, *action_shape), dtype=np.int8)
        self.rewards = np.zeros((0, *reward_shape), dtype=np.float32)
        self.successors = np.zeros((0, *state_shape), dtype=np.int16)
        self.terminals = np.zeros((0, 1), dtype=np.bool)

        #counter to avoid calling len
        self.count = 0

    def __len__(self):
        return self.count

    def add(self, episode_batch):
        '''
        Method to add new samples into the replay buffer.

        :param episode_batch (list): A tuple of np.arrays (states, actions, rewards, successors, terminals).
        '''

        #we concatenate the samples onto the existing tensors
        self.states = np.concatenate([self.states, episode_batch[0].astype('int16')], axis=0)
        self.actions = np.concatenate([self.actions, episode_batch[1].astype('int8')], axis=0)
        self.rewards = np.concatenate([self.rewards, episode_batch[2]], axis=0)
        self.successors = np.concatenate([self.successors, episode_batch[3].astype('int16')], axis=0)
        self.terminals = np.concatenate([self.terminals, episode_batch[4].astype('bool')], axis=0)

        #if the buffer is full or would be overflowing, we delete the episode_length oldest batches from each tensor
        if self.states.shape[0] > self.buffer_size:
            #we calculate the number of elements which would overflow
            overflow_length = self.states.shape[0] - self.buffer_size

            #through slicing we remove the exact number of elements needed to free enough space for the new episode
            self.states = self.states[overflow_length:]
            self.actions = self.actions[overflow_length:]
            self.rewards = self.rewards[overflow_length:]
            self.successors = self.successors[overflow_length:]
            self.terminals = self.terminals[overflow_length:]

            gc.collect()

        #if the buffer is not full, we update the count with the number of added elements
        if self.count != self.buffer_size:
            self.count += episode_batch[0].shape[0]

            if self.count > self.buffer_size:
                self.count = self.buffer_size

    def sample(self, batch_count=1000, batch_size=64):
        '''
        Method to sample elements from the replay buffer.

        :param batch_count (int): The number of batches to sample from the buffer.
        :param batch_size (int): The size of each batch.

        :returns (tf.data.Dataset): A dataset with shuffled batches with specified size of specified length sampled from the buffer.
        '''

        if batch_count*batch_size > self.buffer_size:
            raise IndexError(f'Requested more samples than existing for buffer size: {self.buffer_size}')
                                     
        indices = np.random.choice(np.arange(self.buffer_size), size=int(batch_count*batch_size), replace=False)
        states, actions, rewards, successors, terminals = (
            tf.constant(self.states[(indices)],dtype=tf.float32),
            tf.constant(self.actions[(indices)],dtype=tf.int32),
            tf.constant(self.rewards[(indices)],dtype=tf.float32),
            tf.constant(self.successors[(indices)],dtype=tf.float32),
            tf.constant(self.terminals[(indices)],dtype=tf.bool)
        )
        
        #create a dataset with the requested parameters and returns it
        dataset = tf.data.Dataset.from_tensor_slices((states, actions, rewards, successors, terminals))

        dataset = dataset.batch(batch_size)
        dataset = dataset.take(batch_count)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        del states, actions, rewards, successors, terminals
        gc.collect()
        
        return dataset