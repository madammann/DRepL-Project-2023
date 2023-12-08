import tensorflow as tf
import ast

class ClassifierHead(tf.keras.Model):
    '''
    The final part of either model, a fully-connected network processing a (8,) embedding into a (4,) continous output.
    '''
    
    def __init__(self, layers=1, k_init='glorot_uniform', b_init='zeros'):
        '''
        Method for initializing the model using the subclassing of a tf.keras.Model.
        
        :param layers (int): The number of total Dense layers, excluding the output layer (minimum is 1)
        
        NOTE: There will always be a Dense(8) input and a Dense(4) output layer in addition to the hidden layers.
        '''
        
        super(ClassifierHead, self).__init__()
        
        if layers < 2:
            self.layers = [tf.keras.layers.Dense(units=30, activation='relu', kernel_initializer=k_init, bias_initializer=b_init)]
            
        else:
            self.layers = [tf.keras.layers.Dense(units=30, activation='relu', kernel_initializer=k_init, bias_initializer=b_init) for _ in range(layers)]
        
        self.output_layer = tf.keras.layers.Dense(units=4, activation='sigmoid', kernel_initializer=k_init, bias_initializer=b_init)

    @tf.function
    def __call__(self, x):
        '''
        Method for the feed-forward of data inside the model.
        
        :param x (tf.Tensor): The input to the model in shape (batch, 8).
        
        :returns (tf.Tensor): The final output of the model as tensor of shape (batch, 4).
        '''
        
        for layer in self.layers:
            x = layer(x)

        x = self.output_layer(x)

        return x

class CNNFeatureExtractor(tf.keras.Model):
    '''
    The final part of either model, a fully-connected network processing a (8,) embedding into a (4,) continous output.
    '''
    
    def __init__(self, cnn_depth=1, filters=1, kernel_size=3, k_init='glorot_uniform', b_init='zeros'):
        '''
        ADD
        '''
        
        super(CNNFeatureExtractor, self).__init__()
        
        self.layers = [
            tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=1, padding='same', activation='relu', kernel_initializer=k_init, bias_initializer=b_init) for _ in range(cnn_depth)
        ]
        self.layers += [tf.keras.layers.MaxPooling2D()]
        self.layers = [
            tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=1, padding='same', activation='relu', kernel_initializer=k_init, bias_initializer=b_init) for _ in range(cnn_depth)
        ]
        self.layers += [tf.keras.layers.MaxPooling2D()]

    @tf.function
    def __call__(self, x):
        '''
        :param x (tf.tensor): The input tensor, expected to be of shape (batch, 75, 50, 1).
        '''

        for layer in self.layers:
            x = layer(x)

        return x
    
class MLPFeatureExtractor(tf.keras.Model):
    '''
    ADD
    '''
    
    def __init__(self, layers=1, k_init='glorot_uniform', b_init='zeros', rgb=False):
        '''
        Constructor method for the MLPFeatureExtractor model part.
        
        :param x (ADD): ADD.
        :param x (ADD): ADD.
        :param x (ADD): ADD.
        :param x (ADD): ADD.
        '''
        
        super(MLPFeatureExtractor, self).__init__()
        
        self.flatten = tf.keras.layers.Flatten()
        
        if layers < 1:
            self.layers = [tf.keras.layers.Dense(units=125*(int(rgb)*3), activation='relu', kernel_initializer=k_init, bias_initializer=b_init)]
        
        else:
            self.layers = [tf.keras.layers.Dense(units=125*(int(rgb)*3), activation='relu', kernel_initializer=k_init, bias_initializer=b_init) for _ in range(layers)]
    
    @tf.function
    def __call__(self, x):
        '''
        Call method for the MLPFeatureExtractor model part.
        
        :param x (tf.tensor): The input tensor, expected to be of shape (batch, 75, 50, 1).
        
        :returns (tf.tensor): The output tensor with varying shape based on last Dense layer, defaults (batch, 125).
        '''

        x = self.flatten(x)
        
        for layer in self.layers:
            x = layer(x)

        return x
    
class DeepQNetwork(tf.keras.Model):
    def __init__(self, visual=False, rgb=False, cnn_depth=1, mlp_layers=1, classifier_layers=1, filters=1, kernel_size=3, k_init='glorot_uniform', b_init='zeros'):
        '''
        Constructor method for the DeepQNetworks with MLP and CNN options and MLP classifier head.
        
        :param visual (bool): Boolean flag whether the model should use a CNN as feature extractor or a MLP.
        :param rgb (bool): Boolean flag whether the model expects an RGB image input or a greyscale input.
        :param cnn_depth (): ADD.
        :param mlp_layers (): ADD.
        :param classifier_layers (): ADD.
        :param filters (): ADD.
        :param kernel_size (): ADD.
        :param k_init (): ADD.
        :param b_init (): ADD.
        '''
        
        super(DeepQNetwork, self).__init__()
        
        #we build the model
        self.layers = [ClassifierHead(layers=classifier_layers, k_init=k_init, b_init=b_init)]
        
        #if a visual model, we insert a feature extractor in front of the classifier
        if visual:
            self.layers.insert(0, CNNFeatureExtractor(cnn_depth=1, filters=filters, kernel_size=kernel_size, k_init=k_init, b_init=b_init))
            
        #else insert a MLP for feature extraction in front
        else:
            self.layers.insert(0, MLPFeatureExtractor(layers=mlp_layers, k_init=k_init, b_init=b_init, rgb=rgb))
    
    @tf.function
    def __call__(self, x):
        '''
        ADD
        '''
        
        for layer in self.layers:
            x = layer(x)
            
        return x
    
    def save(self, path : str):
        '''
        ADD
        
        :param path (str): A path with file directory plus affix for the epoch and additional information.
        '''
        
        #storing the model weights
        self.save_weights(path+'weights.h5')
        
        #storing the dictionary of the optimizer state
        with open(path+'optimizer.txt', "w+") as f:
            f.write(str(self.model.optimizer.get_config()))
    
    def load(self, path : str):
        '''
        ADD
        '''
        
        #loading in the model weights and setting built to true
        self.built = True
        self.load_weights(path+'weights.h5')
        
        #loading in the model optimizer state
        with open(path+'optimizer.txt', 'r') as f:
            self.model.optimizer.from_config(ast.literal_eval(f.read()))