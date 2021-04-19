    # coding: utf8
import os
import numpy as np
from pathlib import Path
import json
import sys

from scripts.utils import Collection
from mytools.tools import plot_hist
from mytools.Data import Data
from mytools.DataRelation import DataRelation
from mytools.CustomLayers import MainLSTMBlock, CrossLayer
from mytools.ModelBuilder import ModelBuilder, get_config, get_callbacks

import logging
logger = logging.getLogger(__name__)
#tf.enable_eager_execution()

class ConcreteModel(ModelBuilder):
    """Bert embeddings + WITHOUT char embeddings"""
    CHAR_DICT = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .!?:,\'%-()/$|&;[]"'
    MAX_LEN = 90 
    MAX_LEN_CHAR = 10 
    PRETRAINED = 'bert-base-multilingual-cased'    
    def __init__(self, model_name): 
        super().__init__(model_name)
        print(self.model_dir)
        self.DataProcessor = Data(self.MAX_LEN, self.MAX_LEN_CHAR, self.CHAR_DICT, self.PRETRAINED, char=False)
        self.n_chars = self.DataProcessor.get_n_chars()
        self.n_tags = self.DataProcessor.get_n_tags()      
        self.embedd_dim = self.DataProcessor.get_embedd_dim()
        self.DataRel = DataRelation(self.MAX_LEN)
        self.n_rel = self.DataRel.get_n_rel()  
       
    def build_model(self, config=dict()):
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras import Input
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import concatenate, Dense, TimeDistributed
        
        x = Input(shape=(self.MAX_LEN, self.embedd_dim), dtype='float64')     
        y = MainLSTMBlock(config=config)(x)
        y = TimeDistributed(Dense(self.n_tags, activation="softmax"))(y)

        z = concatenate([x,y]) #change y to argmax(y) ?
        w = CrossLayer()(z)

        w = MainLSTMBlock(config=config)(w)
        w = TimeDistributed(Dense(self.n_rel, activation="softmax"))(w)

        self.Model = Model(x,[y,w])        
        optimizer = Adam(learning_rate = config.pop("learning_rate",0.001))
        self.Model.compile(optimizer, loss="sparse_categorical_crossentropy")

        print(self.Model.summary())
        
                
    def train(self, finput_train: Path, finput_valid: Path, config_path=None, output_name=None):
        """output_dir: ruta donde queremos guardar el modelo entrenado si output_dir = None, ent guardar en model_dir"""
        #TRAIN-VALID SETS
        train_coll, valid_coll = super().get_train_valid_set(finput_train, finput_valid)        
        
        train_set, tokens_train, token_spans_train = self.DataProcessor.process_data(train_coll)
        valid_set, tokens_valid, token_spans_valid = self.DataProcessor.process_data(valid_coll) 

        _, train_rel = self.DataRel.process_input(train_coll, tokens_train, token_spans_train)
        _, valid_rel = self.DataRel.process_input(valid_coll,tokens_valid, token_spans_valid) 

        #MODEL AND FIT CONFIGURATION
        config = get_config(config_path)
 
        #BUILD MODEL FROM SCRATCH
        self.build_model(config)
        
        #CALLBACKS: EARLY STOPPING
        callbacks = get_callbacks(config)

        # #TRAIN
        hist = self.Model.fit( x=train_set[0], y=[np.array(train_set[-1]), np.array(train_rel)]
                                ,validation_data = (valid_set[0],[np.array(valid_set[-1]),np.array(valid_rel)])
                                ,verbose = 1
                                ,epochs = config.pop("epochs",2)
                                ,batch_size = config.pop("batch_size",32)
                                ,callbacks = callbacks
                            )
                            
        logger.info(f"Epoch return by callback: {callbacks[0].stopped_epoch}")                        

        #SAVE TRAINED MODEL
        super().save_model_to(output_name)        
        #PLOT LEARNING CURVE    
        plot_hist(hist, name = os.path.join(self.model_dir, "model_history.txt"))


    def run(self, collection, *args, taskA, taskB, **kargs):
        if not self.Model: 
            self.load_existing_model()               
        if taskA:
            new_set, tokens, tokens_span = self.DataProcessor.process_data(collection)

            prediction = self.Model.predict(x=new_set[0])
            ner_prediction = np.argmax(prediction[0], axis=-1)
            collection = self.DataProcessor.process_predictions(collection, ner_prediction, tokens, tokens_span)  

        if taskB:
            tokens_pairs, _ = self.DataRel.process_input(collection, tokens, tokens_span)
            re_prediction = np.argmax(prediction[1], axis=-1)
            collection = self.DataRel.process_output(collection, re_prediction, tokens_pairs) 

        return collection

