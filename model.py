import tensorflow as tf

class ClassifierHead(tf.keras.Model):
    '''
    The final part of either model, a fully-connected network processing a (8,) embedding into a (4,) continous output.
    '''
    
    def __init__(self, layers=2, units_per_layer=8):
        '''
        Method for initializing the model using the subclassing of a tf.keras.Model.
        
        :param layers (int): The number of hidden layers to use in the classifier head.
        :param units_per_layer (int): The number of units each of the hidden layers of the model should have.
        
        NOTE: There will always be a Dense(8) input and a Dense(4) output layer in addition to the hidden layers.
        '''
        
        super(ClassifierHead, self).__init__()
        
        self.input_layer = tf.keras.layers.Dense(units=8, activation='relu')
        
        self.hidden = [tf.keras.layers.Dense(units=units_per_layer, activation='relu') for _ in range(layers)]
        
        self.output_layer = tf.keras.layers.Dense(units=4, activation='sigmoid')

    @tf.function
    def __call__(self, x):
        '''
        Method for the feed-forward of data inside the model.
        
        :param x (tf.Tensor): The input to the model in shape (batch, 8).
        
        :returns (tf.Tensor): The final output of the model as tensor of shape (batch, 4).
        '''

        x = self.input_layer(x)
        
        for layer in self.hidden:
            x = layer(x)

        x = self.output_layer(x)

        return x