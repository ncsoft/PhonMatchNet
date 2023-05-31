import os, sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.losses import Loss, MeanSquaredError

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

def sequence_cross_entropy(speech_label, text_label, logits, reduction='sum'):
    """
    args
    speech_label        : [B, Ls]
    text_label          : [B, Lt]
    logits              : [B, Lt]
    logits._keras_mask  : [B, Lt]
    """
    # Data pre-processing
    if tf.shape(text_label)[1] > tf.shape(speech_label)[1]:
        speech_label =  tf.pad(speech_label, [[0, 0],[0, tf.shape(text_label)[1] - tf.shape(speech_label)[1]]], 'CONSTANT', constant_values=0)
    elif tf.shape(text_label)[1] < tf.shape(speech_label)[1]:
        speech_label = speech_label[:, :text_label.shape[1]]
    
    # Make paired data between text and speech phonemes
    paired_label = tf.math.equal(text_label, speech_label)
    paired_label = tf.cast(tf.math.logical_and(tf.cast(paired_label, tf.bool), tf.cast(logits._keras_mask, tf.bool)), tf.float32)
    paired_label = tf.reshape(tf.ragged.boolean_mask(paired_label, tf.cast(logits._keras_mask, tf.bool)).flat_values, [-1,1])
    logits = tf.reshape(tf.ragged.boolean_mask(logits, tf.cast(logits._keras_mask, tf.bool)).flat_values, [-1,1])
    
    # Get BinaryCrossEntropy loss
    BCE = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
    loss = BCE(paired_label, logits)
    
    if reduction == 'sum':
        loss = tf.math.divide_no_nan(loss, tf.cast(tf.shape(logits)[0], loss.dtype))
        loss = tf.math.multiply_no_nan(loss, tf.cast(tf.shape(speech_label)[0], loss.dtype))

    return loss

def detection_loss(y_true, y_pred):
    BFC = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
    return(BFC(y_true, y_pred))

class TotalLoss(Loss):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
    
    def __call__(self, y_true, y_pred, reduction='sum'):
        LD = detection_loss(y_true, y_pred)

        return self.weight * LD, LD


class TotalLoss_SCE(Loss):
    def __init__(self, weight=[1.0, 1.0]):
        super().__init__()
        self.weight = weight
    
    def __call__(self, y_true, y_pred, speech_label, text_label, logit, reduction='sum'):
        if self.weight[0] != 0.0:   
            LD = detection_loss(y_true, y_pred)
        else:
            LD = 0
        if self.weight[1] != 0.0:
            LC = sequence_cross_entropy(speech_label, text_label, logit, reduction=reduction)
        else:
            LC = 0
        return self.weight[0] * LD + self.weight[1] * LC, LD, LC