import os, sys
import torch
import numpy as np
import torch.nn as nn

sys.path.append(os.path.dirname(__file__))
import encoder, extractor, discriminator, log_melspectrogram
from utils import sequence_mask

seed = 42
torch.random.manual_seed(seed)
np.random.seed(seed)

class ukws(nn.Module):
    """Base class for user-defined kws mdoel"""
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, speech, text, speech_len=None, text_len=None, ):
        """
        Args:
            speech      : speech feature of shape `(batch, time, *)`
            text        : text embedding of shape `(batch, phoneme, *)`
            speech_len  : length of speech parameter of shape `(batch,)`
            text_len    : length of text parameter of shape `(batch,)`
        """
        raise NotImplementedError

class BaseUKWS(ukws):
    def __init__(self, **kwargs):
        super().__init__()
        embedding = 128
        self.audio_input = kwargs['audio_input']
        self.text_input = kwargs['text_input']
        self.stack_extractor = kwargs['stack_extractor']
        
        _stft={
            'frame_length' : kwargs['frame_length'], 
            'hop_length' : kwargs['hop_length'], 
            'num_mel'  : kwargs['num_mel'] ,
            'sample_rate' : kwargs['sample_rate'],
            'log_mel' : kwargs['log_mel'],
            'lin_to_mel_path' : "./model/lin_to_mel_matrix.npy",    
        }
        if kwargs['audio_input'] == "google_embed":
            input_dim = 96
        else:
            input_dim = kwargs['num_mel']
        _ae = {
            'input_dim' : input_dim,                                
            # [filter, kernel size, stride, padding]
            'conv' : [[embedding, 5, 2, 2], [embedding * 2, 5, 1, 2]],
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
            'num_heads' : 1,
        }
        _dis = {
            'input_dim' : embedding,
            # [unit]
            'gru' : [[embedding],],     
        }

        if self.audio_input == 'both':  # two-stream audio encoder
            self.SPEC = log_melspectrogram.LogMelgramLayer(**_stft)
            self.AE = encoder.EfficientAudioEncoder(downsample=False, **_ae)
        else:                           # single-stream
            if self.audio_input == 'raw':
                self.FEAT = log_melspectrogram.LogMelgramLayer(**_stft)
            elif self.audio_input == 'google_embed':
                pass
            self.AE = encoder.AudioEncoder(**_ae)

        self.TE = encoder.TextEncoder(**_te)
        
        if kwargs['stack_extractor']: 
            self.EXT = extractor.StackExtractor(**_ext)     # self-attention
        else:
            self.EXT = extractor.BaseExtractor(**_ext)      # cross-attention
        
        self.DIS = discriminator.BaseDiscriminator(**_dis)  # Basic keyword discriminator
        
        self.seq_ce_logit = nn.Linear(embedding, 1)         # Additional phoneme discriminator

    def forward(self, speech, text, speech_len=None, text_len=None, verbose=False):
        """
        Args:
            speech      : speech features
                            - if self.audio_input == 'both', shape - `((batch, time, mel), (batch, time/8, 96))`
                            - elif self.audio_input == 'raw', shape - `(batch, time, mel)`
                            - elif self.audio_input == 'google_embed', shape - `(batch, time/8, 96)`
            text        : text embedding of shape `(batch, phoneme)`
            speech_len  : length of speech parameter
                            - if self.audio_input == 'both', shape - `((batch,), (batch,))`
                            - else, shape - `(batch,)`
            text_len    : length of text parameter of shape `(batch,)`
        """
        if self.audio_input == 'both':
            speech, gemb = speech
            s_len, g_len = speech_len
            speech, s_mask = self.SPEC(speech, verbose)
            assert gemb.shape[-1] == 96
            assert speech.shape[1]//8 == gemb.shape[1]
            g_mask = sequence_mask(g_len, gemb.shape[1])
            emb_s, LDN, emb_s_mask = self.AE((speech, gemb), (s_mask, g_mask), verbose)
        else:           
            if self.audio_input == 'raw': 
                speech, s_mask = self.FEAT(speech, verbose)
            elif self.audio_input == 'google_embed':
                speech = speech
                s_mask = sequence_mask(speech_len, speech.shape[1])
            
            emb_s, LDN, emb_s_mask = self.AE(speech, s_mask, verbose)
        
        emb_t, emb_t_mask = self.TE(text, verbose)
        
        attention_output, affinity_matrix, attention_mask, affinity_mask = self.EXT(emb_s, emb_t, emb_s_mask, emb_t_mask, verbose)
        prob, LD = self.DIS(attention_output, attention_mask, verbose)
        
        if self.stack_extractor:
            n_speech = torch.sum(emb_s_mask, dim=-1)
            n_text = torch.sum(emb_t_mask, dim=-1)
            n_total = n_speech + n_text
            # Masking only for the text part: [False, ..., False, True, ..., True, False, ...]
            valid_mask = torch.logical_xor(sequence_mask(n_total, max_length=attention_output.shape[1]), sequence_mask(n_speech, max_length=attention_output.shape[1]))
            indices = torch.masked_fill(torch.cumsum(valid_mask.int(), dim=1), ~valid_mask, 0)
            masked = torch.zeros(attention_output.shape[0], attention_output.shape[1]+1, attention_output.shape[2]).to(attention_output.device)
            masked = torch.scatter(input=masked, dim=1, index=torch.stack([indices for _ in range(attention_output.shape[-1])], dim=-1), src=attention_output)
            valid_attention_output = masked[:,1:torch.max(n_text)+1]
            seq_ce_logit = self.seq_ce_logit(valid_attention_output)[:,:,0]
            seq_ce_logit = nn.functional.pad(seq_ce_logit, (0, emb_t.shape[1] - seq_ce_logit.shape[1]), value=0.)
            seq_ce_logit_mask = emb_t_mask
            seq_ce_logit = torch.nan_to_num(seq_ce_logit) * seq_ce_logit_mask
        
        else:
            seq_ce_logit = self.seq_ce_logit(attention_output)[:,:,0]
            seq_ce_logit_mask = attention_mask
            seq_ce_logit = torch.nan_to_num(seq_ce_logit) * seq_ce_logit_mask
        

        return prob, affinity_matrix, LD, seq_ce_logit, affinity_mask, seq_ce_logit_mask