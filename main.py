import os
import argparse

import pandas as pd

from training import train

# DATA_PATH = './data/'
DATA_PATH = 'D:/experiment_data_DREPL/'

DEFAULT_PARAMS = {
    'batches_per_epoch' : 1000,
    'learning_rate' : 0.001,
    'gamma' : 0.9,
    'epsilon' : 0.9,
    'epsilon_decay' : 0.97,
    'buffer_size_in_batches' : 10000,
    'batch_size' : 16,
    'replay_ratio' : 0.1,
    'polyak_avg_fac' : 0.995
}

STATIC_MLP = {
    'epochs' : 50,
    'visual' : False,
    'rgb' : False,
    'cnn_depth' : 2,
    'mlp_layers' : 2,
    'head_layers' : 1,
    'filters' : 1,
    'kernel_size' : 3,
    'k_init' : 'glorot_uniform',
    'b_init' : 'zeros'
}

STATIC_CNN = {
    'epochs' : 50,
    'visual' : True,
    'rgb' : False,
    'cnn_depth' : 2,
    'mlp_layers' : 2,
    'head_layers' : 1,
    'filters' : 1,
    'kernel_size' : 3,
    'k_init' : 'glorot_uniform',
    'b_init' : 'zeros'
}

parser = argparse.ArgumentParser(description='empty')

parser.add_argument('--showcase',default='False')
parser.add_argument('--iter',default='1')
parser.add_argument('--rgb',default='True')

args = parser.parse_args()

def main(args):
    '''
    ADD
    '''

    cnn_params = {key : val for key, val in list(DEFAULT_PARAMS.items())+list(STATIC_MLP.items())}
    mlp_params = {key : val for key, val in list(DEFAULT_PARAMS.items())+list(STATIC_MLP.items())}

    #if rgb argument is provided as false we change that in the params
    if args.rgb == 'False':
        cnn_params['rgb'], mlp_params['rgb'] = False, False

    #we create an empty dataframe
    columns=['MODEL_NAME','MODEL ID','EPOCH','AVG TD-ERROR']
    df = pd.DataFrame(columns=columns)

    #we load in the data if it exists into the empty dataframe
    if os.path.exists(DATA_PATH+'data.csv'):
        df = pd.read_csv(DATA_PATH+'data.csv', index_col=None)

    #we grab the current latest model id
    current_id = 1
    if len(df) > 0:
        current_id = df['MODEL ID'].max() + 1

    #we train with specified parameters the amount of iterations specified
    for model_id in range(current_id, current_id+int(args.iter)):
        #we do the training and create the dataframe row for the CNN model version
        print('\nTraining CNN Model:\n')
        train(cnn_params, DATA_PATH, 'cnn', model_id, df)

        #reload df
        df = pd.read_csv(DATA_PATH+'data.csv', index_col=None)

        #we do the training and create the dataframe row for the MLP model version
        print('\nTraining MLP Model:\n')
        train(mlp_params, DATA_PATH, 'mlp', model_id, df)

if __name__ == '__main__':
    main(args)
