#from keras.layers.core import Reshape
import tensorflow as tf
from tensorflow.keras.layers import Layer, Lambda, Bidirectional, LSTM, \
                     SpatialDropout1D, Reshape, TimeDistributed, Conv1D, Embedding

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

def all_vs_all_pairs(a):
    a = tf.convert_to_tensor(a) #n_sent x max_len x embedd
    tile_a = tf.tile(tf.expand_dims(a, 2), [1,1,a.get_shape()[1],1])  
    tile_a = tf.expand_dims(tile_a, 3) 
    tile_b = tf.tile(tf.expand_dims(a, 1), [1,a.get_shape()[1],1,1]) 
    tile_b = tf.expand_dims(tile_b, 3) 
    prod = tf.concat([tile_a, tile_b], axis=3) 
    return tf.reshape(prod,(-1,a.get_shape()[1]*a.get_shape()[1],2*a.get_shape()[2]))