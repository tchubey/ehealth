import numpy as np
import tensorflow as tf
#---------------------------------------------------------------#
"""
def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.int8)
  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
"""

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

def scaled_dot_product_attention(q, k, mask=None):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  # scale matmul_qk

  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  mask = tf.cast(mask, tf.float32)
  # add the mask to the scaled tensor.
  #if mask is not None:
  #  scaled_attention_logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  scaled_attention_logits = tf.transpose(scaled_attention_logits, perm=[0, 2, 3, 1])
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k, num_heads)

  #output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
  return attention_weights


def all_vs_all_pairs(a):
    a = tf.convert_to_tensor(a) #n_sent x max_len x embedd
    tile_a = tf.tile(tf.expand_dims(a, 2), [1,1,a.get_shape()[1],1])  
    tile_a = tf.expand_dims(tile_a, 3) 
    tile_b = tf.tile(tf.expand_dims(a, 1), [1,a.get_shape()[1],1,1]) 
    tile_b = tf.expand_dims(tile_b, 3) 
    prod = tf.concat([tile_a, tile_b], axis=3) 
    return tf.reshape(prod,(-1,a.get_shape()[1]*a.get_shape()[1],2*a.get_shape()[2]))

def all_vs_all_pairs_4d(a):
    a = tf.convert_to_tensor(a) #n_sent x max_len x embedd
    tile_a = tf.tile(tf.expand_dims(a, 2), [1,1,a.get_shape()[1],1])  
    tile_b = tf.tile(tf.expand_dims(a, 1), [1,a.get_shape()[1],1,1]) 
    prod = tf.concat([tile_a, tile_b], axis=3) 
    return prod
