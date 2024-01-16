import os
import argparse

import numpy as np
import pandas as pd

from training import train

parser = argparse.ArgumentParser(description='empty')
parser.add_argument('--iter',default='100')
parser.add_argument('--path',default='./hyperparams/hyperparams.csv')
parser.add_argument('--cnn',default='False')

DEFAULT_PARAMS = {
    'batches_per_epoch' : 1000,
    'learning_rate' : 0.001,
    'gamma' : 0.9,
    'epsilon' : 0.9,
    'epsilon_decay' : 0.1,
    'buffer_size_in_batches' : 1000,
    'batch_size' : 64,
    'replay_ratio' : 0.1
}

STATIC_MLP = {
    'epochs' : 1000,
    'visual' : True,
    'rgb' : True,
    'cnn_depth' : 1,
    'mlp_layers' : 1,
    'head_layers' : 1,
    'filters' : 1,
    'kernel_size' : 3,
    'k_init' : 'glorot_uniform',
    'b_init' : 'zeros'
}

STATIC_CNN = {
    'epochs' : 1000,
    'visual' : True,
    'rgb' : True,
    'cnn_depth' : 1,
    'mlp_layers' : 1,
    'head_layers' : 1,
    'filters' : 1,
    'kernel_size' : 3,
    'k_init' : 'glorot_uniform',
    'b_init' : 'zeros'
}

SEARCH_SPACE_BORDERS = {
    'learning_rate' : (0.001, 0.2),
    'gamma' : (0.1, 0.99),
    'epsilon' : (0.0, 0.99),
    'epsilon_decay' : (0.0, 0.99),
    'buffer_size_in_batches' : (1000, 10000),
    'batch_size' : [1, 16, 32, 64],
    'replay_ratio' : (0.01, 0.99)
}

#for search keep initial random buffer(s) in memory/drive between searches

class CustomBayesianHyperparameterOptimizer:
    '''
    ADD
    '''
    
    def __init__(self, training_func, export_path : str, params : dict, restraints : dict, static_params : dict):
        '''
        ADD
        '''
        
        self.export_path = export_path
        self.training_func = training_func
        self.params = params
        self.restraints = restraints
        
        self.data = [] if not os.path.exists(self.export_path) else self.load_dataset()
    
    def find_most_promising_param_set(self):
        '''
        ADD
        '''
        
        pass
    
    def search(self, epoch_depth=1):
        '''
        ADD
        '''
        
        pass
    
    def store_dataset(self):
        '''
        ADD
        '''
        
        pass
    
    def load_dataset(self):
        '''
        ADD
        '''
        
        pass
    
    def __call__(self, iter_lim=100):
        '''
        ADD
        '''
        
        pass
    
    
# we try to find the paramters theta which maximize the objective function

def main(train, path):
    CustomBayesianHyperparameterOptimizer(train, path, params, restraints, static_params)

if __name__ == '__main__':
    pass