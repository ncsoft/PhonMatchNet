import os, sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras import layers

sys.path.append(os.path.dirname(__file__))
import encoder, extractor, discriminator, log_melspectrogram, speech_embedding
from utils import make_feature_matrix as concat_sequence

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

class ukws(Model):
    """Base class for user-defined kws mdoel"""
    
    def __init__(self, name="ukws", **kwargs):
        super(ukws, self).__init__(name=name)

    def call(self, speech, text):
        """
        Args:
            speech  : speech feature of shape `(batch, time)`
            text    : text embedding of shape `(batch, phoneme)`
        """
        raise NotImplementedError

class BaseUKWS(ukws):
    """Base class for user-defined kws mdoel"""
    
    def __init__(self, name="BaseUKWS", **kwargs):
        super(BaseUKWS, self).__init__(name=name)
        embedding=128
        self.audio_input = kwargs['audio_input']
        self.text_input = kwargs['text_input']
        self.stack_extractor = kwargs['stack_extractor']
        
        _stft={
            'frame_length' : kwargs['frame_length'], 
            'hop_length' : kwargs['hop_length'], 
            'num_mel'  : kwargs['num_mel'] ,
            'sample_rate' : kwargs['sample_rate'],
            'log_mel' : kwargs['log_mel'],
        }
        _ae = {
            # [filter, kernel size, stride]
            'conv' : [[embedding, 5, 2], [embedding * 2, 5, 1]],
            # [unit]
            'gru' : [[embedding], [embedding]],
            # fully-connected layer unit
            'fc' : embedding,
            'audio_input' : self.audio_input,
        }
        _te = {
            # fully-connected layer unit
            'fc' : embedding,
            # number of uniq. phonemes
            'vocab' : kwargs['vocab'],
            'text_input' : kwargs['text_input'],
        }
        _ext = {
            # [unit]
            'embedding' : embedding,
        }
        _dis = {
            # [unit]
            'gru' : [[embedding],],
        }
        if self.audio_input == 'both':
            self.SPEC = log_melspectrogram.LogMelgramLayer(**_stft)
            self.EMBD = speech_embedding.GoogleSpeechEmbedder()
            self.AE = encoder.EfficientAudioEncoder(downsample=False, **_ae)
        else:
            if self.audio_input == 'raw':
                self.FEAT = log_melspectrogram.LogMelgramLayer(**_stft)
            elif self.audio_input == 'google_embed':
                self.FEAT = speech_embedding.GoogleSpeechEmbedder()
            self.AE = encoder.AudioEncoder(**_ae)

        self.TE = encoder.TextEncoder(**_te)
        
        if kwargs['stack_extractor']:
            self.EXT = extractor.StackExtractor(**_ext)
        else:
            self.EXT = extractor.BaseExtractor(**_ext)
        
        self.DIS = discriminator.BaseDiscriminator(**_dis)
        
        self.seq_ce_logit = layers.Dense(1, name='sequence_ce')
        
    def call(self, speech, text):
        """
        Args:
            speech      : speech feature of shape `(batch, time)`
            text        : text embedding of shape `(batch, phoneme)`
        """
        if self.audio_input == 'both':
            s = self.SPEC(speech)
            g = self.EMBD(speech)
            emb_s, LDN = self.AE(s, g)
        else:            
            feat = self.FEAT(speech)
            emb_s, LDN = self.AE(feat)
        emb_t = self.TE(text)
        attention_output, affinity_matrix = self.EXT(emb_s, emb_t)
        prob, LD = self.DIS(attention_output)

        if self.stack_extractor:
            n_speech = tf.math.reduce_sum(tf.cast(emb_s._keras_mask, tf.float32), -1)
            n_text = tf.math.reduce_sum(tf.cast(emb_t._keras_mask, tf.float32), -1)
            n_total = n_speech + n_text
            valid_mask = tf.sequence_mask(n_total, maxlen=tf.shape(attention_output)[1], dtype=tf.float32) - tf.sequence_mask(n_speech, maxlen=tf.shape(attention_output)[1], dtype=tf.float32)
            valid_attention_output = tf.ragged.boolean_mask(attention_output, tf.cast(valid_mask, tf.bool)).to_tensor(0.)
            seq_ce_logit = self.seq_ce_logit(valid_attention_output)[:,:,0]
            seq_ce_logit = tf.pad(seq_ce_logit, [[0, 0],[0, tf.shape(emb_t)[1] - tf.shape(seq_ce_logit)[1]]], 'CONSTANT', constant_values=0.)
            seq_ce_logit._keras_mask = emb_t._keras_mask

        else:
            seq_ce_logit = self.seq_ce_logit(attention_output)[:,:,0]
            seq_ce_logit._keras_mask = attention_output._keras_mask
            
        return prob, affinity_matrix, LD, seq_ce_logit