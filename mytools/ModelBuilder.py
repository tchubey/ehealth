  # coding: utf8
import os
import numpy as np
from pathlib import Path
import json
#import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.models import load_model, save_model

from scripts.submit import Algorithm
from scripts.utils import Collection

import logging
logger = logging.getLogger(__name__)
#tf.enable_eager_execution()

class ModelBuilder(Algorithm):
    def __init__(self, model_name, config_path): 
        super(ModelBuilder).__init__()
        self.model_dir = os.path.join('models', model_name)
        self.Model = None
        
        config_path = config_path or os.path.join(self.model_dir,'config.json')
        self.config = get_config(config_path)                
        
        self.MAX_LEN = self.config.get('max_len')
        self.PRETRAINED = self.config.get('pretrained_embeddings')
        self.TOKEN2WORD = self.config.get('token2word')

    def load_existing_model(self, custom_objects=None):
        if Path(self.model_dir).exists():
            try:
                self.Model = load_model(Path(self.model_dir), custom_objects = custom_objects)             
                print("Model was succesfully loaded.")
            except:
                raise ValueError(f"Couldnt load the model from {self.model_dir}.")
        else: raise ValueError("Trained model wasn't found.")
        

    def get_train_valid_set(self, finput_train: Path, finput_valid: Path = None):
        #TRAIN SET
        finput_train = Path(finput_train)
        collection_train = (
            Collection().load_dir(finput_train)
            if finput_train.is_dir()
            else Collection().load(finput_train)
        )        
        #VALIDATION SET
        if finput_valid:        
            finput_valid = Path(finput_valid)
            collection_valid = (
                Collection().load_dir(finput_valid)
                if finput_valid.is_dir()
                else Collection().load(finput_valid)
            )
        else: collection_valid = None
        
        return collection_train, collection_valid
        
    
    def save_model_to(self, output_name=None):
        if output_name: 
            filepath=os.path.join('models',output_name)
        else: filepath=Path(self.model_dir) 
        save_model(self.Model, filepath=filepath)
        save_config(self.config, os.path.join(filepath,'config.json'))      
    
    
    def build_model(self):
        pass 
    
    def train(self):
        pass        
        
    def run(self, *args, taskA, taskB, **kargs):
        pass

def save_config(config, config_path):
    try:
        with open(Path(config_path), 'w') as file:
            json.dump(config, file)
    except:  
        raise ValueError(f'Couldnt save configuration file to {config_path}.')   

def get_config(config_path):
    if config_path:
        try:
            with open(Path(config_path), 'r') as file:
                config = json.load(file)
        except:  
            raise ValueError(f'Couldnt load configuration file from {config_path}.')
    else: config = dict()   
    return config

def get_callbacks(config, checkpoint_filepath = "tmp/model_best"):
    callbacks = []
    callbacks.append(EarlyStopping(
        monitor = 'val_loss',
        mode='min', 
        patience = config.get("patience",10),
        min_delta = config.pop("min_delta",0.01),
        restore_best_weights=True))
    callbacks.append(ModelCheckpoint(
        str(checkpoint_filepath), monitor='loss',
        mode='min', save_best_only=True))     
    callbacks.append(TrainLoggerCallback())
    return callbacks



class TrainLoggerCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logger.info(f'Epoch {epoch}: {logs}')
