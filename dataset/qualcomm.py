import math, os, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.io import wavfile
import tensorflow as tf

from tensorflow.keras.utils import Sequence, OrderedEnqueuer
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences

sys.path.append(os.path.dirname(__file__))
from g2p.g2p_en.g2p import G2p

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

class QualcommKeywordSpeechDataloader(Sequence):
    def __init__(self, 
                 batch_size,
                 fs = 16000,
                 wav_dir='/home/DB/qualcomm_keyword_speech_dataset',
                 target_list=['hey_android', 'hey_snapdragon', 'hi_galaxy', 'hi_lumina'],
                 features='g2p_embed', # phoneme, g2p_embed, both ...
                 shuffle=True,
                 pkl=None,
                 ):
        
        phonemes = ["<pad>", ] + ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
                                    'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 
                                    'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 
                                    'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 
                                    'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0', 
                                    'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 
                                    'UH2', 'UW', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH', 
                                    ' ']
        
        self.p2idx = {p: idx for idx, p in enumerate(phonemes)}
        self.idx2p = {idx: p for idx, p in enumerate(phonemes)}
        
        self.batch_size = batch_size
        self.fs = fs
        self.wav_dir = wav_dir 
        self.target_list = target_list
        self.features = features
        self.shuffle = shuffle
        self.pkl = pkl
        self.nPhoneme = len(phonemes)
        self.g2p = G2p()
                
        self.__prep__()
        self.on_epoch_end()
    
    def __prep__(self):
        self.data = pd.DataFrame(columns=['wav', 'text', 'duration', 'label'])

        if (self.pkl is not None) and (os.path.isfile(self.pkl)):
            print(">> Load dataset from {}".format(self.pkl))
            self.data = pd.read_pickle(self.pkl)
        else:
            print(">> Make dataset from {}".format(self.wav_dir))
            target_dict = {}
            idx = 0
            for target in self.target_list:    
                print(">> Extract from {}".format(target))
                wav_list = [str(x) for x in Path(os.path.join(self.wav_dir, target)).rglob('*.wav')]
                for wav in wav_list:
                    anchor_text = wav.split('/')[-3].lower().replace('_', ' ')
                    duration = float(wavfile.read(wav)[1].shape[-1])/self.fs
                    for comparison_text in self.target_list:
                        comparison_text = comparison_text.replace('_', ' ')
                        label = 1 if anchor_text == comparison_text else 0
                        target_dict[idx] = {
                            'wav': wav,
                            'text': comparison_text,
                            'duration': duration,
                            'label': label
                            }
                        idx += 1
            self.data = self.data.append(pd.DataFrame.from_dict(target_dict, 'index'), ignore_index=True)
    
            # g2p & p2idx by g2p_en package
            print(">> Convert word to phoneme")
            self.data['phoneme'] = self.data['text'].apply(lambda x: self.g2p(re.sub(r"[^a-zA-Z0-9]+", ' ', x)))
            print(">> Convert phoneme to index")
            self.data['pIndex'] = self.data['phoneme'].apply(lambda x: [self.p2idx[t] for t in x])
            print(">> Compute phoneme embedding")
            self.data['g2p_embed'] = self.data['text'].apply(lambda x: self.g2p.embedding(x))

            if (self.pkl is not None) and (not os.path.isfile(self.pkl)):
                self.data.to_pickle(self.pkl)

        # Get longest data
        self.data = self.data.sort_values(by='duration').reset_index(drop=True)
        self.wav_list = self.data['wav'].values
        self.idx_list = self.data['pIndex'].values
        self.emb_list = self.data['g2p_embed'].values
        self.lab_list = self.data['label'].values
        
        # Set dataloader params.
        self.len = len(self.data)
        self.maxlen_t = int((int(self.data['text'].apply(lambda x: len(x)).max() / 10) + 1) * 10)
        self.maxlen_a = int((int(self.data['duration'].values[-1] / 0.5) + 1 ) * self.fs / 2)
                            
    def __len__(self):
        # return total batch-wise length
        return math.ceil(self.len / self.batch_size)

    def _load_wav(self, wav):
        return np.array(wavfile.read(wav)[1]).astype(np.float32) / 32768.0
    
    def __getitem__(self, idx):
        # chunking
        indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        
        # load inputs
        batch_x = [np.array(wavfile.read(self.wav_list[i])[1]).astype(np.float32) / 32768.0 for i in indices]
        if self.features == 'both':
            batch_p = [np.array(self.idx_list[i]).astype(np.int32) for i in indices]
            batch_e = [np.array(self.emb_list[i]).astype(np.float32) for i in indices]
        else:
            if self.features == 'phoneme':
                batch_y = [np.array(self.idx_list[i]).astype(np.int32) for i in indices]
            elif self.features == 'g2p_embed':
                batch_y = [np.array(self.emb_list[i]).astype(np.float32) for i in indices]
        # load outputs
        batch_z = [np.array([self.lab_list[i]]).astype(np.float32) for i in indices]

        # padding and masking
        pad_batch_x = pad_sequences(np.array(batch_x), maxlen=self.maxlen_a, value=0.0, padding='post', dtype=batch_x[0].dtype)
        if self.features == 'both':
            pad_batch_p = pad_sequences(np.array(batch_p), maxlen=self.maxlen_t, value=0.0, padding='post', dtype=batch_p[0].dtype)
            pad_batch_e = pad_sequences(np.array(batch_e), maxlen=self.maxlen_t, value=0.0, padding='post', dtype=batch_e[0].dtype)
        else:
            pad_batch_y = pad_sequences(np.array(batch_y), maxlen=self.maxlen_t, value=0.0, padding='post', dtype=batch_y[0].dtype)
        pad_batch_z = pad_sequences(np.array(batch_z), value=0.0, padding='post', dtype=batch_z[0].dtype)

        if self.features == 'both':
            return pad_batch_x, pad_batch_p, pad_batch_e, pad_batch_z
        else:
            return pad_batch_x, pad_batch_y, pad_batch_z

    def on_epoch_end(self):
        self.indices = np.arange(self.len)
        if self.shuffle == True:
            np.random.shuffle(self.indices)

def convert_sequence_to_dataset(dataloader):
    def data_generator():
        for i in range(dataloader.__len__()):
            if dataloader.features == 'both':
                pad_batch_x, pad_batch_p, pad_batch_e, pad_batch_z = dataloader[i]
                yield pad_batch_x, pad_batch_p, pad_batch_e, pad_batch_z
            else:
                pad_batch_x, pad_batch_y, pad_batch_z = dataloader[i]
                yield pad_batch_x, pad_batch_y, pad_batch_z
    
    if dataloader.features == 'both':
        data_dataset =  tf.data.Dataset.from_generator(data_generator, output_signature=(
            tf.TensorSpec(shape=(None, dataloader.maxlen_a), dtype=tf.float32),
            tf.TensorSpec(shape=(None, dataloader.maxlen_t), dtype=tf.int32),
            tf.TensorSpec(shape=(None, dataloader.maxlen_t, 256), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32),)
        )
    else:
        data_dataset =  tf.data.Dataset.from_generator(data_generator, output_signature=(
            tf.TensorSpec(shape=(None, dataloader.maxlen_a), dtype=tf.float32),
            tf.TensorSpec(shape=(None, dataloader.maxlen_t) if dataloader.features == 'phoneme' else (None, dataloader.maxlen_t, 256),
                        dtype=tf.int32 if dataloader.features == 'phoneme' else tf.float32),
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32),)
        )
    # data_dataset = data_dataset.cache()
    data_dataset = data_dataset.prefetch(1)
    
    return data_dataset

if __name__ == '__main__':
    dataloader = QualcommKeywordSpeechDataloader(2048, pkl='/home/DB/qualcomm_keyword_speech_dataset/qualcomm.pkl', features='g2p_embed')