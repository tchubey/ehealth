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

    def __init__(self, model_name, config_path = None): 
        super().__init__(model_name, config_path)
        # config, config_base, max_len, pretrained, token2word, model_dir, model
        self.DataProcessor = Data(self.MAX_LEN, self.PRETRAINED, self.TOKEN2WORD)
        self.DataRel = DataRelation(self.MAX_LEN)
        self.n_bilou = self.DataProcessor.get_n_bilou()
        self.n_lab = self.DataProcessor.get_n_labels()      
        self.embedd_dim = self.DataProcessor.get_embedd_dim()
        self.n_rel = self.DataRel.get_n_rel()  
       
    def build_model(self):
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras import Input
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import concatenate, Dense, TimeDistributed
        
        input = Input(shape=(self.MAX_LEN, self.embedd_dim), dtype='float32')  
        y = MainGRUBlock(config=self.config)(input)

        bilou = TimeDistributed(Dense(self.n_bilou, activation='softmax'), name = "bilou")(y)
        labels = TimeDistributed(Dense(self.n_lab, activation='softmax'), name = "entities")(y)

        z = tf.argmax(bilou, axis=-1)
        z = tf.one_hot(z, self.n_bilou)

        zz = tf.argmax(labels, axis=-1) 
        zz = tf.one_hot(zz, self.n_lab)

        cc = concatenate([y, z, zz]) 

        w = CrossLayer4D()(cc)
        if self.config.get("dense1",None):
           w = Dense(self.config["dense1"], activation = "relu")(w)
        if self.config.get("dense2",None):
           w = Dense(self.config["dense2"], activation = "relu")(w)
        if self.config.get("dense3",None):
           w = Dense(self.config["dense3"], activation = "relu")(w)

        relations = Dense(self.n_rel, activation="softmax", name="relations")(w)

        output = [bilou, labels, relations]

        self.Model = Model(input, output)        
        optimizer = Adam(learning_rate = self.config.get("learning_rate",0.001))
        self.Model.compile(optimizer, loss_weights={"bilou":1,"entities":1,"relations":1.5},loss=sparse_crossentropy_masked)
        print(self.Model.summary())
        
    #----------- TRAINING -------------#            
    def train(self, finput_train: Path, finput_valid: Path = None, output_name=None, from_scratch = True):
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
        y3_train = self.DataRel.process_input_collection_4d(train_coll, tokens_train, token_spans_train).get("labels")
     
        validation_data = None
        if valid_coll:
            processed_valid = self.DataProcessor.process_input_collection(valid_coll)
            X_valid = processed_valid.get("embeddings")
            tokens_valid = processed_valid.get("tokens")
            token_spans_valid = processed_valid.get("spans")
            y1_valid = processed_valid.get("labels")
            y2_valid = processed_valid.get("bilou")            
            y3_valid = self.DataRel.process_input_collection_4d(valid_coll, tokens_valid, token_spans_valid).get("labels")

            validation_data = (X_valid,[y1_valid, y2_valid, y3_valid])


        #BUILD MODEL FROM SCRATCH
        if from_scratch:
            self.build_model()
        elif not self.Model: 
            self.load_existing_model(custom_objects={"sparse_crossentropy_masked":sparse_crossentropy_masked})  
        
        # TRAIN        
        callbacks = get_callbacks(self.config) #early stopping

        hist = self.Model.fit( x=X_train, y=[y1_train, y2_train, y3_train]
                                ,validation_data = validation_data
                                ,verbose = 1
                                ,epochs = self.config.get("epochs",1)
                                ,batch_size = self.config.get("batch_size",12)
                                ,callbacks = callbacks
                            )                            
        logger.info(f"Epoch return by callback: {callbacks[0].stopped_epoch}")                         

        #SAVE TRAINED MODEL
        super().save_model_to(output_name) 

        #PLOT LEARNING CURVE    
        plot_hist(hist, name = os.path.join(self.model_dir, "model_history"), validation = valid_coll)


    #----------- PREDICTION ----------#
    def run(self, collection,taskA, taskB, *args, **kargs):
        if not self.Model: 
            self.load_existing_model(custom_objects={"sparse_crossentropy_masked":sparse_crossentropy_masked})  
            
        for sentence in collection.sentences:
            logger.debug(sentence)

            processed_sentence = self.DataProcessor.process_input_sentence(sentence)
            X = processed_sentence.get("embeddings")
            tokens = processed_sentence.get("tokens")
            spans = processed_sentence.get("spans")

            prediction = self.Model.predict(x= X)

            if taskA:
                lab_prediction = np.argmax(prediction[0], axis=-1)[0] #[0] para acceder a la primera y unica oracion(sentence)
                bilou_prediction = np.argmax(prediction[1], axis=-1)[0]
                list_of_keypfrases = self.DataProcessor.process_output_sentence(sentence, \
                                        bilou_prediction, lab_prediction, tokens=tokens, spans=spans)  
                logger.debug(list_of_keypfrases)                
                sentence.keyphrases.extend(list_of_keypfrases)

            if taskB:
                re_prediction = np.argmax(prediction[2], axis=-1)[0]
                processed_pairs = self.DataRel.process_input_sentence_4d(sentence, tokens, spans)
                token_pairs = processed_pairs.get("token_pairs")
                list_of_relations = self.DataRel.process_output_sentence(sentence, \
                                        prediction = re_prediction, token_pairs = token_pairs) 

                logger.debug(list_of_relations)
                sentence.relations.extend(list_of_relations)

        return collection

