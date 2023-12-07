import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_datasets as tfds

from datetime import datetime, timedelta
from tqdm import tqdm

class TrainingHandler:
    '''
    ADD
    '''

    def __init__(self, path, episodes=20):
        '''
        ADD
        '''

        #declaring constants
        self.path = path
        self.episodes = episodes

        #declaring variables
        self.runtimes = [60] #initially assume an epoch requires 60mins

        #loading in and preparing the model
        self.model = None

        #initialize buffer
        self.buffer = None

    def sampling(self, batch):
        '''
        ADD
        
        HINT: This function is not optimized for multithreading or multiprocessing yet.
        '''
        
        pass


    def training(self, endtime=None, shutdown=False):
        '''
        ADD

        :param endtime (datetime.datetime):
        :param shutdown (bool):
        '''

        if endtime != None:
            if datetime.now() >= (endtime or (datetime.now()+timedelta(minutes=15))):
                raise ValueError('Must specify an endtime which is at least 15 minutes in the future.')

        #training loop
        for i in range(self.episodes):
            #initially we fill the buffer with random actions
            if i > 0:
                pass
            
            else:
                pass
            
            #do training
            #add
            
def read_schedule():
    '''
    ADD
    '''
    
    pass

def update_schedule():
    '''
    ADD
    '''
    
    pass

def main(args):
    '''
    ADD
    '''
    
    print('\n')

    #we initialize the training handler and read the schedule.txt file
    schedules = read_schedule()
    
    for i, schedule in enumerate(schedules):
        if schedule["complete"] == 'False':
            training_handler = TrainingHandler()
            training_handler.training()
            update_schedule(i)

if __name__ == "__main__":
    main()