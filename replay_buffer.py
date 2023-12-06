import tensorflow as tf

class InMemoryReplayBuffer:
    def __init__(self, state_shape, action_shape, reward_shape, buffer_size=100000):
        '''
        Initialization method for the replay buffer.
        
        :param state_shape (tuple): A tuple representing the shape of the tensor for a state.
        :param action_shape (tuple): A tuple representing the shape of the tensor for an action.
        :param reward_shape (tuple): A tuple representing the shape of the tensor for a reward.
        :param buffer_size (int): The maximum size of the replay buffer, default is 100.000 elements.
        
        NOTE: buffer_size has to be compatible with available memory and dataset.shuffle() limits.
        '''
        
        #maximum size of the buffer
        self.buffer_size = buffer_size
        
        #shapes for each element in the sarst buffer, with state_shape being used for both state and successor state
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.reward_shape = reward_shape
        self.terminal_shape = terminal_shape
        
        #initial empty tensors for each element
        self.states = tf.zeros((0, *state_shape),dtype=tf.float32)
        self.actions = tf.zeros((0, *action_shape),dtype=tf.float32)
        self.rewards = tf.zeros((0, *reward_shape),dtype=tf.float32)
        self.sucessors = tf.zeros((0, *state_shape),dtype=tf.float32)
        self.terminals = tf.zeros((0, 1),dtype=tf.float32)
        
        #counter to avoid calling len
        self.count = 0
    
    def add(self, episode_batch):
        '''
        Method to add new samples into the replay buffer.
        
        :param episode_batch: A tensor with the shape (batch, 5) with dtype=tf.float32.
        '''
        
        #since the length of episodes can be dynamic, we retrieve the length here
        episode_length = episode_batch[0].shape[0]
        
        #if the buffer is full or would be overflowing, we delete the episode_length oldest batches from each tensor
        if episode_length + self.count > self.buffer_size:
            #we calculate the number of elements which would overflow
            overflow_length = self.count + episode_length - self.buffer_size
            
            #through slicing we remove the exact number of elements needed to free enough space for the new episode
            self.states = tf.slice(self.states, [overflow_length, 0], [self.states.shape[0]-overflow_length, *state_shape[1:]])
            self.actions = tf.slice(self.actions, [overflow_length, 0], [self.actions.shape[0]-overflow_length, *self.action_shape[1:]])
            self.rewards = tf.slice(self.rewards, [overflow_length, 0], [self.rewards.shape[0]-overflow_length, *self.reward_shape[1:]])
            self.sucessors = tf.slice(self.sucessors, [overflow_length, 0], [self.sucessors.shape[0]-overflow_length, *self.state_shape[1:]])
            self.terminals = tf.slice(self.terminals, [overflow_length, 0], [self.terminals.shape[0]-overflow_length, *self.terminal_shape[1:]])
        
        #we concatenate the samples onto the existing tensors
        self.states = tf.concat([self.states, episode_batch[0]], axis=0)
        self.actions = tf.concat([self.actions, episode_batch[1]], axis=0)
        self.rewards = tf.concat([self.rewards, episode_batch[2]], axis=0)
        self.successors = tf.concat([self.successors, episode_batch[3]], axis=0)
        self.terminals = tf.concat([self.terminals, episode_batch[4]], axis=0)
        
        #if the buffer is not full, we update the count with the number of added elements
        if self.count != self.buffer_size:
            self.count += episode_length
    
    @tf.function
    def sample(self, batch_count, batch_size):
        '''
        Method to sample elements from the replay buffer.
        
        :param batch_count (int): The number of batches to sample from the buffer.
        :param batch_size (int): The size of each batch.
        :param duplicate (bool): Flag whether duplicate sampling is allowed or not.
        
        :returns (tf.data.Dataset): A dataset with shuffled batches with specified size of specified length sampled from the buffer.
        '''
        
        if batch_count*batch_size > self.buffer_size:
            raise IndexError(f'Requested more samples than existing for buffer size: {self.buffer_size}')
            
        #create a dataset with the requested parameters and returns it
        dataset = tf.data.Dataset.from_tensor_slices((self.states, self.actions, self.rewards, self.sucessors, self.terminals))
        dataset = dataset.shuffle(buffer_size=self.buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset