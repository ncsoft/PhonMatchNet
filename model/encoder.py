import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras import layers

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

class Encoder(Model):
    """Base class for encoders"""
    
    def __init__(self, name="Encoder", **kwargs):
        super(Encoder, self).__init__(name=name)

    def call(self, src, src_len=None):
        """
        Args:
            src     : source of shape `(batch, src_len)`
            src_len : lengths of each source of shape `(batch)`
        """
        raise NotImplementedError

class AudioEncoder(Encoder):
    """Base class for audio encoders"""
    
    def __init__(self, name="BaseAudioEncoder", **kwargs):
        super(AudioEncoder, self).__init__(name=name)
        
        self.crnn = []
        self.stride = 1
        if kwargs['audio_input'] == 'raw':
            for l in kwargs['conv']:
                f, k, s = l
                self.crnn.append(layers.Conv1D(f, k, s, padding='same'))
                self.crnn.append(layers.BatchNormalization())
                self.crnn.append(layers.ReLU())
                self.stride *= s
            for l in kwargs['gru']:
                unit = l
                self.crnn.append(layers.GRU(unit[0], return_sequences=True))

        self.dense = layers.Dense(kwargs['fc'])
        self.act = layers.LeakyReLU()
            
    def call(self, src):
        """
        Args:
            src         : source of shape `(batch, time, feature)`
            src_len     : lengths of each source of shape `(batch)`
        """
        # keep the batch mask
        mask_flag = 'mask' in vars(src)
        if mask_flag:
            mask = src.mask[:,::self.stride]

        x = src
        for layer in self.crnn:
            # [B, T, F] -> [B, T/2, Conv1D] -> [B, T/2, GRU]
            if isinstance(layer, layers.GRU):
                if mask_flag:
                    x = layer(x, mask=mask)
                else:
                    x = layer(x)
            else:
                x = layer(x)
        # [B, T/2, Dense]
        x = self.dense(x)
        LD = x
        x = self.act(x)
        if mask_flag:
            x._keras_mask = mask
        
        return x, LD
    
class EfficientAudioEncoder(Encoder):
    """Efficient encoder class for audio encoders"""
    
    def __init__(self, name="EfficientAudioEncoder", downsample=True, **kwargs):
        super(EfficientAudioEncoder, self).__init__(name=name)
        self.downsample = downsample
        self.layer = []
        
        if self.downsample:
            for _ in range(2):
                self.layer.append(layers.Conv1D(kwargs['fc'], 5, 2, padding='same'))
                self.layer.append(layers.BatchNormalization())
                self.layer.append(layers.ReLU())
                self.layer.append(layers.MaxPool1D(pool_size=2, strides=2, padding='valid'))
            self.layer = self.layer[:-1]
        else:
            self.layer.append(layers.Conv1D(kwargs['fc'], 3, 2, padding='same'))
            self.layer.append(layers.BatchNormalization())
            self.layer.append(layers.ReLU())
            self.layer.append(layers.Conv1D(kwargs['fc'], 3, 1, padding='same'))
            self.layer.append(layers.BatchNormalization())
            self.layer.append(layers.ReLU())
            self.deConv = layers.Conv1DTranspose(kwargs['fc'], 5, 4)
        
        self.dense = layers.Dense(kwargs['fc'])
        self.act = layers.LeakyReLU()
            
    def call(self, specgram, embed):
        """
        Args:
            embed       : google speech embedding of shape `(batch, time / 8, 96)`
            specgram    : log mel-spectrogram of shape `(batch, time, mel)`
        """
        # keep the batch mask
        mask_flag = 'mask' in vars(embed)
        
        if mask_flag:
            if self.downsample:
                mask = specgram.mask[:,::8]
            else:
                mask = specgram.mask[:,::2]

        x = specgram
        for l in self.layer:
            # [B, T, F] -> [B, T/8, dense] or [B, T/2, dense]
            x = l(x)

        LD = x
        
        # [B, T/8, dense] or [B, T/2, dense]
        if self.downsample:
            y = self.act(self.dense(embed))
            
            # Summation two embedding
            x += tf.pad(y, [[0, 0],[0, tf.shape(x)[1] - tf.shape(y)[1]],[0, 0]], 'CONSTANT', constant_values=0.0)
        else:
            y = self.act(self.deConv(embed))
            if tf.shape(x)[1] > tf.shape(y)[1]:
                x += tf.pad(y, [[0, 0],[0, tf.shape(x)[1] - tf.shape(y)[1]],[0, 0]], 'CONSTANT', constant_values=0.0)
            elif tf.shape(x)[1] < tf.shape(y)[1]:
                x += y[:, :x.shape[1], :]

        if mask_flag:
            x._keras_mask = mask
            LD._keras_mask = mask
            
        return x, LD

class TextEncoder(Encoder):
    """Base class for text encoders"""
    
    def __init__(self, name="BaseTextEncoder", **kwargs):
        super(TextEncoder, self).__init__(name=name)
        
        self.features = kwargs['text_input']
        if self.features == 'phoneme':
            self.mask = tf.keras.layers.Masking(mask_value=0, input_shape=(None,))
            vocab = tf.convert_to_tensor(kwargs['vocab'], dtype=tf.int32)
            self.one_hot = layers.Lambda(lambda x: tf.one_hot(x, vocab), dtype=tf.int32, name='ont_hot')
        elif self.features == 'g2p_embed':
            self.mask = tf.keras.layers.Masking(mask_value=0, input_shape=(None, 256))
        self.dense = layers.Dense(kwargs['fc'])
        self.act = layers.LeakyReLU()

    def call(self, src):
        """
        Args:
            src         : phoneme token of shape `(batch, phoneme)`
            src_len     : lengths of each source of shape `(batch)`
        """
        # [B, phoneme] -> [B, phoneme, embedding]
        mask = self.mask(src)
        if self.features == 'phoneme':
            x = self.one_hot(src)
        elif self.features == 'g2p_embed':
            x = src
            mask = mask[:,:,0]
        x = self.act(self.dense(x))
        x._keras_mask = tf.cast(mask, tf.bool)

        return x