    # coding: utf8
import os
import numpy as np
from pathlib import Path
import sys

#from scripts.utils import Collection
from mytools.tools import plot_hist
from mytools.Data import Data
from mytools.DataRelation import DataRelation
from mytools.CustomLayers import *
from mytools.ModelBuilder import ModelBuilder, get_config, get_callbacks

from mytools.tools import sparse_crossentropy_masked
np.set_printoptions(threshold=sys.maxsize)

import logging
logger = logging.getLogger(__name__)
#tf.enable_eager_execution()

class ConcreteModel(ModelBuilder):
    """Bert embeddings + WITHOUT char embeddings"""
    MAX_LEN = 20 
    PRETRAINED = 'bert-base-multilingual-cased' 

    def __init__(self, model_name): 
        super().__init__(model_name)
        self.DataProcessor = Data(self.MAX_LEN, self.PRETRAINED)
        self.n_bilou = self.DataProcessor.get_n_bilou()
        self.n_tags = self.DataProcessor.get_n_labels()      
        self.embedd_dim = self.DataProcessor.get_embedd_dim()
        self.DataRel = DataRelation(self.MAX_LEN)
        self.n_rel = self.DataRel.get_n_rel()  
       
    def build_model(self, config=dict()):
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras import Input
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import concatenate, Dense, TimeDistributed, Softmax
        
        x = Input(shape=(self.MAX_LEN, self.embedd_dim), dtype='float32')  

        y = MainLSTMBlock(config=config)(x)#, mask = mask_bool)

        y_bilou = TimeDistributed(Dense(self.n_bilou, activation='softmax', name = "bilou"))(y)
        y_tags = TimeDistributed(Dense(self.n_tags, activation='softmax', name = "entities"))(y)

        z = tf.argmax(y_tags, axis=-1) #ValueError: A `Concatenate` layer requires inputs with matching shapes except for the concat axis. Got inputs shapes: [(None, 30, 768), (None, 30)]
        z = tf.one_hot(z, self.n_tags)

        inputs = [x]

        cc = concatenate([y,z]) 

        w = CrossLayer4D()(cc)
        w = Dense(200, activation = "relu")(w)
        w = Dense(100, activation = "relu")(w)
        w = Dense(self.n_rel, activation="softmax", name="relations")(w)

        outputs = [y_bilou,y_tags,w]

        self.Model = Model(inputs,outputs)        
        optimizer = Adam(learning_rate = config.get("learning_rate",0.001))
        self.Model.compile(optimizer, loss=sparse_crossentropy_masked)#, loss_weights={"entities":1,"relations":1} )

        print(self.Model.summary())
        
    #----------- TRAINING -------------#            
    def train(self, finput_train: Path, finput_valid: Path = None, config_path=None, output_name=None, from_scratch = True):
        """output_dir: ruta donde queremos guardar el modelo entrenado si output_dir = None, ent guardar en model_dir"""
        #TRAIN-VALID SETS
        train_coll, valid_coll = super().get_train_valid_set(finput_train, finput_valid)        
        
        # ENTITIES
        processed_train = self.DataProcessor.process_input_collection(train_coll)
        X_train = processed_train.get("embeddings")
        tokens_train = processed_train.get("tokens")
        token_spans_train = processed_train.get("spans")
        y1_train = processed_train.get("labels")
        y2_train = processed_train.get("bilou")

        # RELATIONS
        z_train = self.DataRel.process_input_collection_4d(train_coll, tokens_train, token_spans_train).get("labels")

        validation_data = None
        if valid_coll:
            processed_valid = self.DataProcessor.process_input_collection(valid_coll)
            X_valid = processed_valid.get("embeddings")
            tokens_valid = processed_valid.get("tokens")
            token_spans_valid = processed_valid.get("spans")
            y1_valid = processed_valid.get("labels")
            y2_valid = processed_train.get("bilou")            
            z_valid = self.DataRel.process_input_collection_4d(valid_coll, tokens_valid, token_spans_valid).get("labels")

            validation_data = ([X_valid],[y1_valid, y2_valid, z_valid])
        
        #MODEL AND FIT CONFIGURATION
        config = get_config(config_path)
 
        #BUILD MODEL FROM SCRATCH
        if from_scratch:
            self.build_model(config)
        elif not self.Model: 
                self.load_existing_model(custom_objects={"sparse_crossentropy_masked":sparse_crossentropy_masked})  
        
        # TRAIN        
        callbacks = get_callbacks(config) #early stopping
        hist = self.Model.fit( x=[X_train], y=[y1_train, y2_train, z_train]
                                ,validation_data = validation_data
                                ,verbose = 1
                                ,epochs = config.pop("epochs",1)
                                ,batch_size = config.pop("batch_size",12)
                                ,callbacks = callbacks
                            )                            
        logger.info(f"Epoch return by callback: {callbacks[0].stopped_epoch}")                         

        #SAVE TRAINED MODEL
        super().save_model_to(output_name) 

        #PLOT LEARNING CURVE    
        plot_hist(hist, name = os.path.join(self.model_dir, "model_history"), validation = valid_coll)


    #----------- PREDICTION ----------#
    def run(self, collection, *args, taskA, taskB, **kargs):
        if not self.Model: 
            self.load_existing_model(custom_objects={"sparse_crossentropy_masked":sparse_crossentropy_masked})  
            
        for sentence in collection.sentences:
            logger.debug(sentence)

            processed_sentence = self.DataProcessor.process_input_sentence(sentence)
            X1 = processed_sentence.get("embeddings")
            tokens = processed_sentence.get("tokens")
            spans = processed_sentence.get("spans")

            prediction = self.Model.predict(x=[X1])

            if taskA:
                lab_prediction = np.argmax(prediction[0], axis=-1)[0] #[0] para acceder a la primera y unica oracion(sentence)
                bilou_prediction = np.argmax(prediction[1], axis=-1)[0]

                list_of_keypfrases = self.DataProcessor.process_output_sentence(sentence, \
                                        bilou_prediction, lab_prediction, tokens=tokens, spans=spans)  
                logger.debug(list_of_keypfrases)                
                sentence.keyphrases.extend(list_of_keypfrases)

            if taskB:
                re_prediction = np.argmax(prediction[2], axis=-1)[0]

                processed_pairs = self.DataRel.process_input_sentence(sentence, tokens, spans)
                token_pairs = processed_pairs.get("token_pairs")
                list_of_relations = self.DataRel.process_output_sentence(sentence, \
                                        prediction = re_prediction, token_pairs = token_pairs) 

                logger.debug(list_of_relations)
                sentence.relations.extend(list_of_relations)

            #print(collection.sentences[0])
        return collection

