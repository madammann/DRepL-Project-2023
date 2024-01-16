import tensorflow as tf
import ast

class ModelHead(tf.keras.Model):
    '''
    The ModelHead part of either model, a fully-connected network processing an embedding into a (4,) continous output.
    '''

    def __init__(self, layers=1, k_init='glorot_uniform', b_init='zeros'):
        '''
        Constructor method for the ModelHead model part.

        :param layers (int): The number of total Dense layers, excluding the output layer, minimum is one.
        :param k_init (str): The chosen TensorFlow kernel initialisation method.
        :param b_init (str): The chosen TensorFlow bias initialisation method.

        NOTE: There will always be a Dense(4) output layer in addition to the layers.
        '''

        super(ModelHead, self).__init__()

        if layers < 2:
            self.model_layers = [tf.keras.layers.Dense(units=30, activation='relu', kernel_initializer=k_init, bias_initializer=b_init)]

        else:
            self.model_layers = [tf.keras.layers.Dense(units=30, activation='relu', kernel_initializer=k_init, bias_initializer=b_init) for _ in range(layers)]

        self.q_val_layer = tf.keras.layers.Dense(units=4, activation='sigmoid', kernel_initializer=k_init, bias_initializer=b_init)

    @tf.function
    def __call__(self, x):
        '''
        Method for the feed-forward of data inside the model.

        :param x (tf.Tensor): The input embedding to the model ModelHead in shape (batch, n) for n feature dimension.

        :returns (tuple): The final output of the model as tuple of q_value and policy (val : tf.tensor(batch, 1), pol : tf.tensor(batch, 4)).
        '''

        for layer in self.model_layers:
            x = layer(x)

        return self.q_val_layer(x)

class CNNFeatureExtractor(tf.keras.Model):
    '''
    The Convolutional Feature Extractor to be used before the PolicyHead of the model.
    '''

    def __init__(self, cnn_depth=1, filters=1, kernel_size=3, k_init='glorot_uniform', b_init='zeros'):
        '''
        Constructor method for the CNNFeatureExtractor model part.

        :param cnn_depth (int): The number of Conv2D layers before each of the two Max Pooling operations.
        :param filters (int): The number of filters each Conv2D layer should have.
        :param kernel_size (int or tuple): The kernel_size each Conv2D layer should have.
        :param k_init (str): The chosen TensorFlow kernel initialisation method.
        :param b_init (str): The chosen TensorFlow bias initialisation method.
        '''

        super(CNNFeatureExtractor, self).__init__()

        self.model_layers = [
            tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=1, padding='same', activation='relu', kernel_initializer=k_init, bias_initializer=b_init) for _ in range(cnn_depth)
        ]
        self.model_layers += [tf.keras.layers.MaxPooling2D()]
        self.model_layers = [
            tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=1, padding='same', activation='relu', kernel_initializer=k_init, bias_initializer=b_init) for _ in range(cnn_depth)
        ]
        self.model_layers += [tf.keras.layers.MaxPooling2D()]

    @tf.function
    def __call__(self, x):
        '''
        :param x (tf.tensor): The input tensor, expected to be of shape (batch, 75, 50, 1).

        :returns (tf.Tensor): The output of the feature extractor as tensor of shape (batch, 18, 12,1).
        '''

        for layer in self.model_layers:
            x = layer(x)

        return x

class MLPFeatureExtractor(tf.keras.Model):
    '''
    The Multi Layer Perceptron Feature Extractor to be used before the PolicyHead of the model.
    '''

    def __init__(self, layers=1, k_init='glorot_uniform', b_init='zeros', rgb=False):
        '''
        Constructor method for the MLPFeatureExtractor model part.

        :param layers (int): The number of Dense layers in the feature extractor, minimum 1.
        :param k_init (str): The chosen TensorFlow kernel initialisation method.
        :param b_init (str): The chosen TensorFlow bias initialisation method.
        :param rgb (bool): A Boolean to indicate whether the unit count of the Dense layers has to be trippled (True), or not.
        '''

        super(MLPFeatureExtractor, self).__init__()

        self.flatten = tf.keras.layers.Flatten()

        if layers < 1:
            selfmodel_layers.model_layers = [tf.keras.layers.Dense(units=125*(int(rgb)*3), activation='relu', kernel_initializer=k_init, bias_initializer=b_init)]

        else:
            self.model_layers = [tf.keras.layers.Dense(units=125*(int(rgb)*3), activation='relu', kernel_initializer=k_init, bias_initializer=b_init) for _ in range(layers)]

    @tf.function
    def __call__(self, x):
        '''
        Call method for the MLPFeatureExtractor model part.

        :param x (tf.tensor): The input tensor, expected to be of shape (batch, 75, 50, 1).

        :returns (tf.tensor): The output tensor with varying shape based on last Dense layer, defaults (batch, 125) or (batch, 125*3) for RGB.
        '''

        x = self.flatten(x)

        for layer in self.model_layers:
            x = layer(x)

        return x

class DeepQNetwork(tf.keras.Model):
    def __init__(self, optimizer, visual=False, rgb=True, cnn_depth=1, mlp_layers=1, head_layers=1, filters=1, kernel_size=3, k_init='glorot_uniform', b_init='zeros'):
        '''
        Constructor method for the DeepQNetworks with MLP and CNN options and MLP policy head.

        :param optimizer: The optimizer to be used during training by the model.
        :param visual (bool): Boolean flag whether the model should use a CNN as feature extractor or a MLP.
        :param rgb (bool): Boolean flag whether the model expects an RGB image input or a greyscale input.
        :param cnn_depth (int): The number of Conv2D layers before each of the two Max Pooling operations.
        :param mlp_layers (int): The number of Dense layers in the feature extractor, minimum is one.
        :param head_layers (int): The number of total Dense layers in the classifier head, excluding the output layer, minimum is one.
        :param filters (int): The number of filters each Conv2D layer should have.
        :param kernel_size (int or tuple): The kernel_size each Conv2D layer should have.
        :param k_init (str): The chosen TensorFlow kernel initialisation method.
        :param b_init (str): The chosen TensorFlow bias initialisation method.

        NOTE: Certain parameters do not work together, or rather have no effect with visual set to True or False.
        '''

        super(DeepQNetwork, self).__init__()

        #we store the config parameters
        self.config_parameters = {
            'optimizer' : str(type(optimizer)).replace("'",'"'),
            'visual' : visual,
            'rgb' : rgb,
            'cnn_depth' : cnn_depth,
            'mlp_layers' : mlp_layers,
            'head_layers' : head_layers,
            'filters' : filters,
            'kernel_size' : kernel_size,
            'k_init' : k_init,
            'b_init' : b_init
        }

        #we store the optimizer in a variable
        self.optimizer = optimizer

        #we build the model
        self.model_layers = [ModelHead(layers=head_layers, k_init=k_init, b_init=b_init)]

        #if a visual model, we insert a feature extractor in front of the classifier
        if visual:
            self.model_layers.insert(0, CNNFeatureExtractor(cnn_depth=1, filters=filters, kernel_size=kernel_size, k_init=k_init, b_init=b_init))

        #else insert a MLP for feature extraction in front
        else:
            self.model_layers.insert(0, MLPFeatureExtractor(layers=mlp_layers, k_init=k_init, b_init=b_init, rgb=rgb))

    @tf.function
    def __call__(self, x):
        '''
        Call method for the MLPFeatureExtractor model part.

        :param x (tf.tensor): The input tensor, expected to be of shape (batch, 75, 50, c) for image input, with c color channels.

        :returns (tf.Tensor): The final output of the model as tensor of shape (batch, 4).
        '''

        for layer in self.model_layers:
            x = layer(x)

        return x

    def save(self, path : str, path_affix=""):
        '''
        Method for saving the model and the optimizer state.
        Assumes a path to a directory and will write a default name for the model weights file and optimizer.
        For a more specific model name and iteration, please provide a path affix.

        :param path (str): A path with file directory plus affix for the epoch and additional information.
        :param path_affix (str): An additional string to be used inside the directory at path as an affix to the "weights.h5" and "optimizer.txt".
        '''

        #storing the model weights
        self.save_weights(path+path_affix+'weights.h5')

        #storing the dictionary of the optimizer state and config parameters together
        with open(path+path_affix+'optimizer.txt', "w+") as f:
            f.write(str({key : val for key, val in list(self.config_parameters.items())+list(self.optimizer.get_config().items())}))

    def load(self, path : str, path_affix=""):
        '''
        Method for loading the model and the optimizer state.
        Assumes the path to lead to a directory with the weights.h5 and the optimizer.txt inside it.
        The additional affix is used to load in a specific file from the directory if multiple are present.

        :param path (str): A path with file directory plus affix for the epoch and additional information.
        :param path_affix (str): An additional string to be used inside the directory at path as an affix to the "weights.h5" and "optimizer.txt".
        '''

        #loading in the model weights and setting built to true
        self.built = True
        self.__call__(tf.random.normal((1, 75, 50, 3 if self.config_parameters['rgb'] else 1), dtype=tf.float32))
        self.load_weights(path+path_affix+'weights.h5')

        #loading in the model optimizer state and config parameters by splitting them
        with open(path+path_affix+'optimizer.txt', 'r') as f:
            dictionary = ast.literal_eval(f.read())
            self.config_parameters = {key : val for key, val in dictionary.items() if key in self.config_parameters.keys()}
            self.optimizer.from_config({key : val for key, val in dictionary.items() if not key in self.config_parameters.keys()})
