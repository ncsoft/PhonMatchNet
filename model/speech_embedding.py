import os, sys, time
import tensorflow as tf
from tensorflow.keras.models import Model

class GoogleSpeechEmbedder(Model):
    def __init__(self, name="google_embedding", **kwargs):
        super(GoogleSpeechEmbedder, self).__init__(name=name)
        
        self._embeddingModel = tf.saved_model.load(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'google_speech_embedding'), tags=[]).signatures["default"]
        self.window = 12400
        self.shift = 1280
        self.pre_padding = self.window - self.shift
        self.non_trainable_weights.append(self._embeddingModel.variables)

    def __call__(self, speech):
        batch = speech.shape[0]
        mask = tf.keras.layers.Masking(mask_value=0.0)(tf.expand_dims(speech, -1))._keras_mask
        speech = tf.concat([tf.zeros([speech.shape[0], self.pre_padding]), speech], -1)

        assert speech.shape[-1] > self.window, 'Input speech length must over 880 samples'
        est_end = int(1 + (speech.shape[-1] - self.window) // self.shift)
        trim = int(speech.shape[-1] % self.shift)
        speech = speech[:, :-trim]
        speech = tf.concat([tf.reshape(speech, [1, -1]), tf.zeros([1, self.pre_padding])], -1)
        emb = tf.reshape(self._embeddingModel(speech)['default'], [batch, -1, 1, 96])[:, :est_end, :, :]
        emb = tf.squeeze(emb, axis=2)
        emb.mask = mask[:,self.shift-1::self.shift]
        emb._keras_mask = emb.mask

        return emb