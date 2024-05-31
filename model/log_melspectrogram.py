import torch
import torch.nn as nn
import numpy as np

class LogMelgramLayer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.log_mel = kwargs['log_mel']
        self.num_fft = 1 << (kwargs['frame_length'] - 1).bit_length()
        self.hop_length = kwargs['hop_length']
        self.frame_length = kwargs['frame_length']

        num_freqs = (self.num_fft // 2) + 1
        """
        lin_to_mel_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=kwargs['num_mel'],
            num_spectrogram_bins=num_freqs,
            sample_rate=kwargs['sample_rate'],
            lower_edge_hertz=80,
            upper_edge_hertz=kwargs['sample_rate']/2,
        )
        """
        lin_to_mel_matrix = torch.from_numpy(np.load(kwargs['lin_to_mel_path'])) 
        # Pre-computed matrix! 
        # Configs: 'frame_length' : 400, 'num_mel'  : 40, 'sample_rate' : 16000,
        # Shape: `(num_frequencies, num_mel_bins)`

        self.lin_to_mel_matrix = lin_to_mel_matrix

    def forward(self, input, verbose=False):
        """
        Args:
            input (tensor): Batch of mono waveform, shape: (None, N)

        Returns:
            log_melgrams (tensor): Batch of log mel-spectrograms, shape: (None, num_frame, mel_bins)

        """
        window = torch.hann_window(self.frame_length).to(input.device)
        stfts = torch.stft(
            input, 
            self.num_fft, 
            win_length=self.frame_length, 
            hop_length=self.hop_length, 
            window=window, 
            return_complex=True,
        )
        mag_stfts = torch.abs(stfts)

        melgrams = torch.tensordot(             # (batch_size, time_steps, num_mel_bins)
            torch.square(mag_stfts).transpose(1, 2), 
            self.lin_to_mel_matrix.to(input.device), 
            dims=([2], [0])
            ) 
        melgrams_mask = torch.sum((melgrams != 0.0), dim=-1) > 0
        melgrams = torch.nan_to_num(melgrams) * melgrams_mask.unsqueeze(-1)

        if self.log_mel:
            log_melgrams = torch.log10(melgrams + 1e-7)
            log_melgrams_mask = torch.sum((log_melgrams != -7.0), dim=-1) > 0
            log_melgrams = torch.nan_to_num(log_melgrams) * log_melgrams_mask.unsqueeze(-1)
            return log_melgrams, log_melgrams_mask
        else:
            return melgrams, melgrams_mask