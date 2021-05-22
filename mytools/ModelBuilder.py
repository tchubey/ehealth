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
    def __init__(self, model_name): 
        super(ModelBuilder).__init__()
        self.model_dir = os.path.join('models', model_name)
        self.Model = None
        
    def load_existing_model(self, custom_objects=None):
        if Path(self.model_dir).exists():
            try:
                self.Model = load_model(Path(self.model_dir ), custom_objects = custom_objects)
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
            save_model(self.Model, filepath=os.path.join('models',output_name))
        else: save_model(self.Model, filepath=Path(self.model_dir))        
    
    
    def build_model(self, config):
        pass 
    
    def train(self, finput_train: Path, finput_valid: Path, config_path=None, output_name=None):
        """output_dir: ruta donde queremos guardar el modelo entrenado si output_dir = None, ent guardar en model_dir"""
        pass        
        
    def run(self, collection, *args, taskA, taskB, **kargs):
        pass

def get_config(config_path=None):
    if config_path:
        try:
            with open(Path(config_path), 'r') as file:
                config = json.load(file)
        except: 
            config = dict()  
            print(f'Couldnt load configuration file from {config_path}. Default configuration will be used.')
    else: config = dict()   
    return config

def get_callbacks(config, checkpoint_filepath = "tmp/model_best"):
    callbacks = []

    callbacks.append(EarlyStopping(
        monitor = 'val_loss',
        mode='min', 
        patience = config.pop("patience",3),
        min_delta = config.pop("min_delta",0.01),
        restore_best_weights=True))
    callbacks.append(ModelCheckpoint(
        str(checkpoint_filepath), monitor='val_loss',
        mode='min', save_best_only=True))     
    callbacks.append(TrainLoggerCallback())
    return callbacks

class TrainLoggerCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logger.info(f'Epoch {epoch}: {logs}')
