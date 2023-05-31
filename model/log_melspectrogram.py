from syslog import LOG_DAEMON
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers

class LogMelgramLayer(Model):
    def __init__(self, name="mel_specgram", **kwargs):
        if kwargs['log_mel']:
            super(LogMelgramLayer, self).__init__(name="log_mel_specgram",)
        else:
            super(LogMelgramLayer, self).__init__(name=name,)
            
        self.log_mel = kwargs['log_mel']
        num_fft = 1 << (kwargs['frame_length'] - 1).bit_length()
        self.hop_length = kwargs['hop_length']
        self.frame_length = kwargs['frame_length']

        num_freqs = (num_fft // 2) + 1
        lin_to_mel_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=kwargs['num_mel'],
            num_spectrogram_bins=num_freqs,
            sample_rate=kwargs['sample_rate'],
            lower_edge_hertz=80,
            upper_edge_hertz=kwargs['sample_rate']/2,
        )

        self.lin_to_mel_matrix = lin_to_mel_matrix
        self.non_trainable_weights.append(self.lin_to_mel_matrix)

    def call(self, input):
        """
        Args:
            input (tensor): Batch of mono waveform, shape: (None, N)

        Returns:
            log_melgrams (tensor): Batch of log mel-spectrograms, shape: (None, num_frame, mel_bins, channel=1)

        """

        def _tf_log10(x):
            numerator = tf.math.log(x)
            denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
            return numerator / denominator
      
        # tf.signal.stft seems to be applied along the last axis
        stfts = tf.signal.stft(
            input, frame_length=self.frame_length, frame_step=self.hop_length
        )
        mag_stfts = tf.abs(stfts)

        melgrams = tf.tensordot(tf.square(mag_stfts), self.lin_to_mel_matrix, axes=[2, 0])
        melgrams.mask = layers.Masking(mask_value=0.0)(melgrams)._keras_mask

        if self.log_mel:
            log_melgrams = _tf_log10(melgrams + tf.keras.backend.epsilon())
            log_melgrams.mask = layers.Masking(mask_value=-7.0)(log_melgrams)._keras_mask
            return log_melgrams
        else:
            return melgrams