#from keras.layers.core import Reshape
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Lambda, Bidirectional, LSTM, Dense, \
                     SpatialDropout1D, Reshape, TimeDistributed, \
                     Conv1D, MaxPooling1D, GlobalMaxPooling1D, Embedding, Flatten

from transformers import DistilBertConfig, TFDistilBertModel
from mytools.tools_layers import *

class MainLSTMBlock(Layer):
    def __init__(self, config = dict()):
        super(MainLSTMBlock, self).__init__()
        self.spatial_droput = config.get("spatial_droput",0.3)
        self.bilstm_units = config.get("bilstm_units",30)
        self.bilstm_dropout = config.get("bilstm_dropout",0.5)
        self.spatial = SpatialDropout1D(self.spatial_droput)
        self.bilstm = Bidirectional(LSTM(units=self.bilstm_units, return_sequences=True,
                                        recurrent_dropout=self.bilstm_dropout))
        self.bilstm_2 = Bidirectional(LSTM(units=self.bilstm_units, return_sequences=True,
                                        recurrent_dropout=self.bilstm_dropout))

    def call(self, inputs):
        x = self.spatial(inputs)
        x = self.bilstm(x)
        x = self.bilstm_2(x)
        return x



class CharCNNBlock(Layer):
    def __init__(self, num_char, max_len_char, config = dict()):
        super(CharCNNBlock, self).__init__()
        self.embedd_dim = config.get("embedding_dim",10)
        self.filters = config.get("filters",50)
        self.kernel = config.get("kernel_size",max_len_char)
                                        
        self.embedding = TimeDistributed(Embedding(input_dim=num_char, output_dim=self.embedd_dim,
                                            input_length=max_len_char))

        self.conv = Conv1D(filters=self.filters, kernel_size=self.kernel, activation="relu")

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.conv(x)        
        x = Reshape((x.shape[1]*x.shape[2], x.shape[3]))(x)
        return x

class mainCNNBlock(Layer):
    def __init__(self, config = dict()):
        super(mainCNNBlock, self).__init__()
        self.filters = config.get("filters",30)
        self.kernel = config.get("kernel_size",10)
        self.pool_size = config.get("pool_size",10)
        self.units = config.get("dense",50)
        # recibe (None,90x90,768x2)     
        # (None,90x90,50)                           
        self.conv = Conv1D(filters=self.filters, kernel_size=self.kernel, activation="relu", padding='same')
        self.maxpool = GlobalMaxPooling1D()#(pool_size = self.pool_size, padding='same')
        self.flatten = Flatten()
        self.dense = Dense(self.units, activation = "relu")
    def call(self, inputs):
        x = self.conv(inputs)   
        #x = self.maxpool(x)     
        #x = Reshape((x.shape[1]*x.shape[2], x.shape[3]))(x)
        return x    
        
class CharLSTMBlock(Layer):
    def __init__(self, num_char, max_len_char, config=dict()):
        super(CharLSTMBlock, self).__init__()
        self.embedd_dim = config.get("embedding_dim",10)
        self.lstm_units = config.get("lstm_units",30)
        self.lstm_droput = config.get("lstm_droput",0.5)
                                        
        self.embedding = TimeDistributed(Embedding(input_dim=num_char, output_dim=self.embedd_dim,
                                            input_length=max_len_char))        
        self.lstm = TimeDistributed(LSTM(units= self.lstm_units, return_sequences=False,
                                            recurrent_dropout= self.lstm_droput))
                                            
    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.lstm(x)
        return x

class CrossLayer(Layer):
    def __init__(self):
        super(CrossLayer, self).__init__()
        self.cross = Lambda(all_vs_all_pairs, name="cross_layer")

    def call(self, x):
        return self.cross(x)

class CrossLayer4D(Layer):
    def __init__(self):
        super(CrossLayer4D, self).__init__()
        self.cross = Lambda(all_vs_all_pairs_4d, name="cross_layer")

    def call(self, x):
        return self.cross(x)
"""
class Distil(Layer):
    def __init__(self, dim=None):
        super(Distil, self).__init__()
        self.configuration = DistilBertConfig(n_layers = 2, n_heads = 2, dim = dim) #dim??
        self.distil = TFDistilBertModel(config = self.configuration)
        self.dense = Dense(units=50)

    def call(self, x):
        x = self.distil.call({'inputs_embeds': x}).last_hidden_state 
        x = self.dense(x)  
        return x     
"""

class PaddingMask(tf.keras.layers.Layer):
    def __init__(self, num_heads):
        super(PaddingMask, self).__init__()
        self.num_heads = num_heads
                  
    def call(self, x):
        #x = create_padding_mask(x)
        #x: mask where 0 if PAD
        x = x[:, tf.newaxis, tf.newaxis, :]
        padding_mask = tf.matmul(x, x, transpose_a=True) # ! quiero 1x0 = 1
        ones = tf.ones_like(padding_mask)
        padding_mask = tf.subtract(ones, padding_mask) #invierto el valor
        #padding_mask = tf.bitwise.invert(padding_mask)

        padding_mask = tf.tile(padding_mask, multiples = tf.constant([1,self.num_heads,1,1]))
        return padding_mask


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, max_len, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.max_len = max_len
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        
        #v = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
   
    def call(self, x, mask=None):
        batch_size = tf.shape(x)[0]
        d_model = x.shape[2]

        x = x + positional_encoding(self.max_len, d_model)
 
        q = self.wq(x)  # (batch_size, seq_len, d_model)
        k = self.wk(x)  # (batch_size, seq_len, d_model)
        #v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        #v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        attention_weights = scaled_dot_product_attention(q, k, mask)

        return attention_weights
        
