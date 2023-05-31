import os, sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras import layers

sys.path.append(os.path.dirname(__file__))
from utils import make_adjacency_matrix, make_feature_matrix

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

class Extractor(Model):
    """Base class for extractors"""
    
    def __init__(self, name="Extractor", **kwargs):
        super(Extractor, self).__init__(name=name)

    def call(self, emb_s, emb_t, emb_s_len=None, emb_t_len=None):
        """
        Args:
            emb_s   : speech embedding of shape `(batch, time, embedding)`
            emb_t   : text embedding of shape `(batch, phoneme, embedding)`
            emb_s_len   : length of speech embedding of shape `(B,)`
            emb_t_len   : length of text embedding of shape `(B,)`
        """
        raise NotImplementedError

class BaseExtractor(Extractor):
    """Base class for pattern extractor"""
    
    def __init__(self, name="BaseExtractor", **kwargs):
        super(BaseExtractor, self).__init__(name=name)
        self.embedding = kwargs['embedding']
        self.attn = layers.MultiHeadAttention(num_heads=1, key_dim=self.embedding)
            
    def call(self, emb_s, emb_t):
        """
        Args:
            emb_s   : speech embedding of shape `(batch, time, embedding)`
            emb_t   : text embedding of shape `(batch, phoneme, embedding)`
            emb_s_len   : length of speech embedding of shape `(B,)`
            emb_t_len   : length of text embedding of shape `(B,)`
            * Query - text, Key,Value - speech *
        """
        Q = emb_t
        V = emb_s
        
        # [B, Tt, m], [B, Tt, Ta] notation followed Learning Audio-Text Agreement for Open-vocabulary Keyword Spotting
        if ('_keras_mask' in vars(Q)) and ('_keras_mask' in vars(V)):
            if Q._keras_mask is None:
                attn_mask = None
            else:
                attn_mask = tf.expand_dims(tf.cast(Q._keras_mask, tf.int32), -1) * tf.expand_dims(tf.cast(V._keras_mask, tf.int32), 1)
                    
        else:
            attn_mask = None

        attention_output, affinity_matrix = self.attn(Q, V, 
                                                return_attention_scores=True,
                                                attention_mask = attn_mask
                                                )
        if self.attn._num_heads == 1:
            affinity_matrix = affinity_matrix[:,0,:,:]
        if attn_mask is not None:
            affinity_matrix._keras_mask = attn_mask

        if attn_mask is not None:
            attention_output._keras_mask = Q._keras_mask
                            
        return attention_output, affinity_matrix
        

class StackExtractor(Extractor):
    """Self-attention based pattern extractor"""
    
    def __init__(self, name="StackExtractor", **kwargs):
        super(StackExtractor, self).__init__(name=name)
        self.embedding = kwargs['embedding']
        self.attn = layers.MultiHeadAttention(num_heads=1, key_dim=self.embedding)
            
    def call(self, emb_s, emb_t):
        """
        Args:
            emb_s   : speech embedding of shape `(batch, time, embedding)`
            emb_t   : text embedding of shape `(batch, phoneme, embedding)`
            * Query - text, Key,Value - speech *
        """
        Q = make_feature_matrix(emb_s, emb_s._keras_mask, emb_t, emb_t._keras_mask)
        V = Q
        attn_mask = make_adjacency_matrix(emb_s._keras_mask, emb_t._keras_mask)

        attention_output, affinity_matrix = self.attn(Q, V, 
                                                return_attention_scores=True,
                                                attention_mask = attn_mask
                                                )
        if self.attn._num_heads == 1:
            affinity_matrix = affinity_matrix[:,0,:,:]
        if attn_mask is not None:
            affinity_matrix._keras_mask = attn_mask

        if attn_mask is not None:
            attention_output._keras_mask = attn_mask[:,:,0]
            
        return attention_output, affinity_matrix