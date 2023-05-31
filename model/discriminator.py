import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras import layers

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

class Discriminator(Model):
    """Base class for discriminators"""
    
    def __init__(self, name="Discriminator", **kwargs):
        super(Discriminator, self).__init__(name=name)

    def call(self, src, src_len=None):
        """
        Args:
            src     : source of shape `(batch, src_len)`
            src_len : lengths of each source of shape `(batch)`
        """
        raise NotImplementedError

class BaseDiscriminator(Discriminator):
    """Base class for discriminators"""
    
    def __init__(self, name="BaseDiscriminator", **kwargs):
        super(BaseDiscriminator, self).__init__(name=name)
        self.gru = []
        for i, l in enumerate(kwargs['gru']):
            unit = l
            if i == len(kwargs['gru']) - 1:
                self.gru.append(layers.GRU(unit[0], return_sequences=False))
            else:
                self.gru.append(layers.GRU(unit[0], return_sequences=True))
        self.dense = layers.Dense(1)
        self.act = layers.Lambda(lambda x: tf.keras.activations.sigmoid(x), name='sigmoid')

        
    def call(self, src, src_len=None):
        """
        Args:
            src         : source of shape `(batch, time, feature)`
            src_len     : lengths of each source of shape `(batch)`
        """
        x = src
        for layer in self.gru:
            # [B, Tt, m] -> [B, embedding]
            if '_keras_mask' in vars(src):
                x = layer(x, mask=tf.cast(src._keras_mask, tf.bool))
            else:
                x = layer(x)
        # [B, 1]
        x = self.dense(x)
        return self.act(x), x