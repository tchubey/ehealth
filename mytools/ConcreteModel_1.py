    # coding: utf8
import os
import numpy as np
from pathlib import Path
import json

from scripts.submit import Algorithm
from scripts.utils import Collection, Keyphrase, Relation, Sentence, ENTITIES
#from mytools.tools import plot_hist
from mytools.Data import Data
from mytools.CustomLayers import MainLSTMBlock, CharCNNBlock, CharLSTMBlock
from mytools.ModelBuilder import ModelBuilder, get_config, get_callbacks

import logging
logger = logging.getLogger(__name__)
#tf.enable_eager_execution()

class ConcreteModel(ModelBuilder):
    """Bert embeddings + without char embedd + CRF layer (instead of Dense)"""
    CHAR_DICT = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .!?:,\'%-()/$|&;[]"'
    MAX_LEN = 90 
    MAX_LEN_CHAR = 10 
    PRETRAINED = 'bert-base-multilingual-cased'    
    def __init__(self, model_name): 
        """
        ##model_dir##: si queremos cargar un modelo existente para seguir entrenando o para predecir
        Si no quiero seguir entrenando, sino que quiero crear un modelo nuevo, entonces debo hacer train(..,train_new=True)
        """
        super().__init__(model_name)
        self.DataProcessor = Data(self.MAX_LEN, self.MAX_LEN_CHAR, self.CHAR_DICT, self.PRETRAINED, char=False)
        self.n_chars = self.DataProcessor.get_n_chars()
        self.n_tags = self.DataProcessor.get_n_tags()
        self.embedd_dim = self.DataProcessor.get_embedd_dim()        

    def build_model(self, config):
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras import Input
        from tensorflow.keras.models import Model
        from tf2crf import CRF, ModelWithCRFLoss
        
        x = Input(shape=(self.MAX_LEN, self.embedd_dim), dtype='float64')        

        y = MainLSTMBlock(config=config)(x)
        y = CRF(units=self.n_tags)(y) 
        self.Model = Model(x,y)        
        self.Model = ModelWithCRFLoss(self.Model, sparse_target=True)
        
        optimizer = Adam(learning_rate = config.pop("learning_rate",0.001))
        self.Model.compile(optimizer)
        
        #print(self.Model.summary())
        
 
    def train(self, finput_train: Path, finput_valid: Path, config_path=None, output_name=None):
        """output_dir: ruta donde queremos guardar el modelo entrenado si output_dir = None, ent guardar en model_dir"""
        #TRAIN-VALID SETS
        train_coll, valid_coll = super().get_train_valid_set(finput_train, finput_valid)        
        train_set, _, _ = self.DataProcessor.process_data(train_coll)
        valid_set, _, _ = self.DataProcessor.process_data(valid_coll) 
        
        #MODEL AND FIT CONFIGURATION
        config = get_config(config_path)
 
        #BUILD MODEL FROM SCRATCH (y si quiero seguir entrenadno?!)
        self.build_model(config)
        
        #CALLBACKS: EARLY STOPPING
        callbacks = get_callbacks(config)
        
        # #TRAIN
        hist = self.Model.fit( x=train_set[:-1], y=train_set[-1],
                                validation_data = (valid_set[:-1],valid_set[-1]),
                                verbose = 1,
                                epochs = config.pop("epochs",3), 
                                batch_size = config.pop("batch_size",32), 
                                callbacks = callbacks)
                            
        #logger.debug(f"Epoch return by callback: {callbacks[0].stopped_epoch}")                        
        
        #PLOT LEARNING CURVE    
        #plot_hist(hist)
        f = open(os.path.join(self.model_dir, "model_history.txt"), "a")
        f.write(hist.history)
        f.close()

        #SAVE TRAINED MODEL
        super().save_model_to(output_name)
        

        
    def run(self, collection, *args, taskA, taskB, **kargs):
        if not self.Model: 
            self.load_existing_model()                
        if taskA:
            new_set, tokens, tokens_span = self.DataProcessor.process_data(collection)
            
            prediction = self.Model.predict(x=new_set[:-1])
            collection = self.DataProcessor.process_predictions(collection, prediction, tokens, tokens_span)  

        if taskB:
            pass 
        return collection

