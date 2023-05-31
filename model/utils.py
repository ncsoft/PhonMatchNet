import math
import tensorflow as tf
from tensorflow.keras import layers

def make_adjacency_matrix(speech_mask, text_mask):
    """
    args
    speech_mask : [B, Ls]
    text_mask   : [B, Lt]
    """
    # [B, L] -> [B]
    n_speech = tf.math.reduce_sum(tf.cast(speech_mask, tf.float32), -1)
    n_text = tf.math.reduce_sum(tf.cast(text_mask, tf.float32), -1)
    n_node = n_speech + n_text
    max_len = tf.math.reduce_max(n_node)
    # [B] -> [B, max_len] -> [B, max_len, 1] * [B, 1, max_len]-> [B, max_len, max_len]
    mask = tf.sequence_mask(n_node, maxlen=max_len, dtype=tf.float32)
    mask = tf.expand_dims(mask, -1) * tf.expand_dims(mask, 1)
    # Make upper triangle matrix for adj. matrix
    adjacency_matrix = tf.linalg.band_part(mask, -1, 0)
    
    return adjacency_matrix

def make_feature_matrix(speech_features, speech_mask, text_features, text_mask):
    """
    args
    speech_features : [B, Ls, F]
    speech_mask     : [B, Ls]
    text_features   : [B, Lt, F]
    text_mask       : [B, Lt]
    """
    # Data pre-processing
    speech_mask = tf.cast(speech_mask, tf.float32)
    text_mask = tf.cast(text_mask, tf.float32)
    speech_seq_mask = tf.tile(tf.expand_dims(speech_mask, -1), tf.constant([1, 1, speech_features.shape[-1]], tf.int32))
    text_seq_mask = tf.tile(tf.expand_dims(text_mask, -1), tf.constant([1, 1, text_features.shape[-1]], tf.int32))
    speech_features *= speech_seq_mask
    text_features *= text_seq_mask
    
    # Concatenate two feature matrix along time axis
    feature_matrix = tf.concat([speech_features, text_features], axis=1)
    feature_mask = tf.concat([speech_mask, text_mask], axis=-1)
    
    # Gather valid data using mask : tensor -> ragged tensor -> tensor
    return tf.ragged.boolean_mask(feature_matrix, tf.cast(feature_mask, tf.bool)).to_tensor(0.)