import math, os, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import Levenshtein
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

class LibriPhraseDataloader(Sequence):
    def __init__(self, 
                 batch_size,
                 fs = 16000,
                 wav_dir='/home/DB/LibriPhrase_diffspk_all',
                 noise_dir='/home/DB/noise',
                 csv_dir='/home/DB/LibriPhrase/data',
                 train_csv = ['train_100h', 'train_360h'],
                 test_csv = ['train_500h',],
                 types='both', # easy, hard
                 features='g2p_embed', # phoneme, g2p_embed, both ...
                 train=True,
                 shuffle=True,
                 pkl=None,
                 edit_dist=False,
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
        self.csv_dir = csv_dir
        self.noise_dir = noise_dir
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.types = types
        self.features = features
        self.train = train
        self.shuffle = shuffle
        self.pkl = pkl
        self.edit_dist = edit_dist
        self.nPhoneme = len(phonemes)
        self.g2p = G2p()
        
        self.__prep__()
        self.on_epoch_end()
    
    def __prep__(self):
        if self.train:
            print(">> Preparing noise DB")
            noise_list = [str(x) for x in Path(self.noise_dir).rglob('*.wav')]
            self.noise = np.array([])
            for noise in noise_list:
                fs, data = wavfile.read(noise)
                assert fs == self.fs, ">> Error : Un-match sampling freq.\n{} -> {}".format(noise, fs)
                data = data.astype(np.float32) / 32768.0
                data = (data / np.max(data)) * 0.5
                self.noise = np.append(self.noise, data)
            
        self.data = pd.DataFrame(columns=['wav_label', 'wav', 'text', 'duration', 'label', 'type'])

        if (self.pkl is not None) and (os.path.isfile(self.pkl)):
            print(">> Load dataset from {}".format(self.pkl))
            self.data = pd.read_pickle(self.pkl)
        else:
            for db in self.train_csv if self.train else self.test_csv:
                csv_list = [str(x) for x in Path(self.csv_dir).rglob('*' + db + '*word*')]
                for n_word in csv_list:
                    print(">> processing : {} ".format(n_word))
                    df = pd.read_csv(n_word)
                    # Split train dataset to match & unmatch case
                    anc_pos = df[['anchor_text', 'anchor', 'anchor_text', 'anchor_dur']]
                    anc_neg = df[['anchor_text', 'anchor', 'comparison_text', 'anchor_dur', 'target', 'type']]
                    com_pos = df[['comparison_text', 'comparison', 'comparison_text', 'comparison_dur']]
                    com_neg = df[['comparison_text', 'comparison', 'anchor_text', 'comparison_dur', 'target', 'type']]
                    anc_pos.columns = ['wav_label', 'anchor', 'anchor_text', 'anchor_dur']
                    com_pos.columns = ['wav_label', 'comparison', 'comparison_text', 'comparison_dur']
                    anc_pos['label'] = 1
                    anc_pos['type'] = df['type']
                    com_pos['label'] = 1
                    com_pos['type'] = df['type']
                    # Concat
                    self.data = self.data.append(anc_pos.rename(columns={y: x for x, y in zip(self.data.columns, anc_pos.columns)}), ignore_index=True)
                    self.data = self.data.append(anc_neg.rename(columns={y: x for x, y in zip(self.data.columns, anc_neg.columns)}), ignore_index=True)
                    self.data = self.data.append(com_pos.rename(columns={y: x for x, y in zip(self.data.columns, com_pos.columns)}), ignore_index=True)
                    self.data = self.data.append(com_neg.rename(columns={y: x for x, y in zip(self.data.columns, com_neg.columns)}), ignore_index=True)

            # Append wav directory path
            self.data['wav'] = self.data['wav'].apply(lambda x: os.path.join(self.wav_dir, x))
            # g2p & p2idx by g2p_en package
            print(">> Convert word to phoneme")
            self.data['phoneme'] = self.data['text'].apply(lambda x: self.g2p(re.sub(r"[^a-zA-Z0-9]+", ' ', x)))
            print(">> Convert speech word to phoneme")
            self.data['wav_phoneme'] = self.data['wav_label'].apply(lambda x: self.g2p(re.sub(r"[^a-zA-Z0-9]+", ' ', x)))
            print(">> Convert phoneme to index")
            self.data['pIndex'] = self.data['phoneme'].apply(lambda x: [self.p2idx[t] for t in x])
            print(">> Convert speech phoneme to index")
            self.data['wav_pIndex'] = self.data['wav_phoneme'].apply(lambda x: [self.p2idx[t] for t in x])
            print(">> Compute phoneme embedding")
            self.data['g2p_embed'] = self.data['text'].apply(lambda x: self.g2p.embedding(x))
            print(">> Calucate Edit distance ratio")
            self.data['dist'] = self.data.apply(lambda x: Levenshtein.ratio(re.sub(r"[^a-zA-Z0-9]+", ' ', x['wav_label']), re.sub(r"[^a-zA-Z0-9]+", ' ', x['text'])), axis=1)

            if (self.pkl is not None) and (not os.path.isfile(self.pkl)):
                self.data.to_pickle(self.pkl)
        
        # Masking dataset type
        if self.types == 'both':
            pass
        elif self.types == 'easy':
            self.data = self.data.loc[self.data['type'] == 'diffspk_easyneg']
        elif self.types == 'hard':
            self.data = self.data.loc[self.data['type'] == 'diffspk_hardneg']

        # Get longest data
        self.data = self.data.sort_values(by='duration').reset_index(drop=True)
        self.wav_list = self.data['wav'].values
        self.idx_list = self.data['pIndex'].values
        self.sIdx_list = self.data['wav_pIndex'].values
        self.emb_list = self.data['g2p_embed'].values
        self.lab_list = self.data['label'].values
        if self.edit_dist:
            self.dist_list = self.data['dist'].values
        
        # Set dataloader params.
        self.len = len(self.data)
        self.maxlen_t = int((int(self.data['text'].apply(lambda x: len(x)).max() / 10) + 1) * 10)
        self.maxlen_a = int((int(self.data['duration'].values[-1] / 0.5) + 1 ) * self.fs / 2)
        self.maxlen_l = int((int(self.data['wav_label'].apply(lambda x: len(x)).max() / 10) + 1) * 10)
                            
    def __len__(self):
        # return total batch-wise length
        return math.ceil(self.len / self.batch_size)

    def _load_wav(self, wav):
        return np.array(wavfile.read(wav)[1]).astype(np.float32) / 32768.0
    
    def _mixing_snr(self, clean, snr=[5, 15]):
        def _cal_adjusted_rms(clean_rms, snr):
            a = float(snr) / 20
            noise_rms = clean_rms / (10**a) 
            return noise_rms

        def _cal_rms(amp):
            return np.sqrt(np.mean(np.square(amp), axis=-1))
        
        start = np.random.randint(0, len(self.noise)-len(clean))
        divided_noise = self.noise[start: start + len(clean)]
        clean_rms = _cal_rms(clean)
        noise_rms = _cal_rms(divided_noise)
        adj_noise_rms = _cal_adjusted_rms(clean_rms, np.random.randint(snr[0], snr[1]))
        
        adj_noise_amp = divided_noise * (adj_noise_rms / (noise_rms + 1e-7)) 
        noisy = clean + adj_noise_amp
        
        if np.max(noisy) > 1:
            noisy = noisy / np.max(noisy)
        
        return noisy
    
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
        batch_l = [np.array(self.sIdx_list[i]).astype(np.int32) for i in indices]
        batch_t = [np.array(self.idx_list[i]).astype(np.int32) for i in indices]
        if self.edit_dist:
            batch_d = [np.array([self.dist_list[i]]).astype(np.float32) for i in indices]

        # padding and masking
        pad_batch_x = pad_sequences(np.array(batch_x), maxlen=self.maxlen_a, value=0.0, padding='post', dtype=batch_x[0].dtype)
        if self.features == 'both':
            pad_batch_p = pad_sequences(np.array(batch_p), maxlen=self.maxlen_t, value=0.0, padding='post', dtype=batch_p[0].dtype)
            pad_batch_e = pad_sequences(np.array(batch_e), maxlen=self.maxlen_t, value=0.0, padding='post', dtype=batch_e[0].dtype)
        else:
            pad_batch_y = pad_sequences(np.array(batch_y), maxlen=self.maxlen_t, value=0.0, padding='post', dtype=batch_y[0].dtype)
        pad_batch_z = pad_sequences(np.array(batch_z), value=0.0, padding='post', dtype=batch_z[0].dtype)
        pad_batch_l = pad_sequences(np.array(batch_l), maxlen=self.maxlen_l, value=0.0, padding='post', dtype=batch_l[0].dtype)
        pad_batch_t = pad_sequences(np.array(batch_t), maxlen=self.maxlen_t, value=0.0, padding='post', dtype=batch_t[0].dtype)
        if self.edit_dist:
            pad_batch_d = pad_sequences(np.array(batch_d), value=0.0, padding='post', dtype=batch_d[0].dtype)
        
        # Noisy option
        if self.train:
            batch_x_noisy = [self._mixing_snr(x) for x in batch_x]
            pad_batch_x_noisy = pad_sequences(np.array(batch_x_noisy), maxlen=self.maxlen_a, value=0.0, padding='post', dtype=batch_x_noisy[0].dtype)
        
        if self.train:
            if self.features == 'both':
                return pad_batch_x, pad_batch_x_noisy, pad_batch_p, pad_batch_e, pad_batch_z, pad_batch_l, pad_batch_t
            else:
                return pad_batch_x, pad_batch_x_noisy, pad_batch_y, pad_batch_z, pad_batch_l, pad_batch_t
        else:
            if self.features == 'both':
                if self.edit_dist:
                    return pad_batch_x, pad_batch_p, pad_batch_e, pad_batch_z, pad_batch_d
                else:
                    return pad_batch_x, pad_batch_p, pad_batch_e, pad_batch_z
            else:
                if self.edit_dist:
                    return pad_batch_x, pad_batch_y, pad_batch_z, pad_batch_d
                else:
                    return pad_batch_x, pad_batch_y, pad_batch_z

    def on_epoch_end(self):
        self.indices = np.arange(self.len)
        if self.shuffle == True:
            np.random.shuffle(self.indices)

def convert_sequence_to_dataset(dataloader):
    def data_generator():
        for i in range(dataloader.__len__()):
            if dataloader.train:
                if dataloader.features == 'both':
                    pad_batch_x, pad_batch_x_noisy, pad_batch_p, pad_batch_e, pad_batch_z, pad_batch_l, pad_batch_t = dataloader[i]
                    yield pad_batch_x, pad_batch_x_noisy, pad_batch_p, pad_batch_e, pad_batch_z, pad_batch_l, pad_batch_t
                else:
                    pad_batch_x, pad_batch_x_noisy, pad_batch_y, pad_batch_z, pad_batch_l, pad_batch_t = dataloader[i]
                    yield pad_batch_x, pad_batch_x_noisy, pad_batch_y, pad_batch_z, pad_batch_l, pad_batch_t
            else:
                if dataloader.features == 'both':
                    if dataloader.edit_dist:
                        pad_batch_x, pad_batch_p, pad_batch_e, pad_batch_z, pad_batch_d = dataloader[i]
                        yield pad_batch_x, pad_batch_p, pad_batch_e, pad_batch_z, pad_batch_d
                    else:
                        pad_batch_x, pad_batch_p, pad_batch_e, pad_batch_z = dataloader[i]
                        yield pad_batch_x, pad_batch_p, pad_batch_e, pad_batch_z
                else:
                    if dataloader.edit_dist:
                        pad_batch_x, pad_batch_y, pad_batch_z, pad_batch_d = dataloader[i]
                        yield pad_batch_x, pad_batch_y, pad_batch_z, pad_batch_d
                    else:
                        pad_batch_x, pad_batch_y, pad_batch_z = dataloader[i]
                        yield pad_batch_x, pad_batch_y, pad_batch_z
    
    if dataloader.train:
        if dataloader.features == 'both':
            data_dataset =  tf.data.Dataset.from_generator(data_generator, output_signature=(
                tf.TensorSpec(shape=(None, dataloader.maxlen_a), dtype=tf.float32),
                tf.TensorSpec(shape=(None, dataloader.maxlen_a), dtype=tf.float32),
                tf.TensorSpec(shape=(None, dataloader.maxlen_t), dtype=tf.int32),
                tf.TensorSpec(shape=(None, dataloader.maxlen_t, 256), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None, dataloader.maxlen_l), dtype=tf.int32),
                tf.TensorSpec(shape=(None, dataloader.maxlen_t), dtype=tf.int32),)
            )
        else:
            data_dataset =  tf.data.Dataset.from_generator(data_generator, output_signature=(
                tf.TensorSpec(shape=(None, dataloader.maxlen_a), dtype=tf.float32),
                tf.TensorSpec(shape=(None, dataloader.maxlen_a), dtype=tf.float32),
                tf.TensorSpec(shape=(None, dataloader.maxlen_t) if dataloader.features == 'phoneme' else (None, dataloader.maxlen_t, 256),
                            dtype=tf.int32 if dataloader.features == 'phoneme' else tf.float32),
                tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None, dataloader.maxlen_l), dtype=tf.int32),
                tf.TensorSpec(shape=(None, dataloader.maxlen_t), dtype=tf.int32),)
            )
    else:
        if dataloader.features == 'both':
            if dataloader.edit_dist:
                data_dataset =  tf.data.Dataset.from_generator(data_generator, output_signature=(
                    tf.TensorSpec(shape=(None, dataloader.maxlen_a), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, dataloader.maxlen_t), dtype=tf.int32),
                    tf.TensorSpec(shape=(None, dataloader.maxlen_t, 256), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, 1), dtype=tf.float32),)
                )
            else:
                data_dataset =  tf.data.Dataset.from_generator(data_generator, output_signature=(
                    tf.TensorSpec(shape=(None, dataloader.maxlen_a), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, dataloader.maxlen_t), dtype=tf.int32),
                    tf.TensorSpec(shape=(None, dataloader.maxlen_t, 256), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, 1), dtype=tf.float32),)
                )
        else:
            if dataloader.edit_dist:
                data_dataset =  tf.data.Dataset.from_generator(data_generator, output_signature=(
                    tf.TensorSpec(shape=(None, dataloader.maxlen_a), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, dataloader.maxlen_t) if dataloader.features == 'phoneme' else (None, dataloader.maxlen_t, 256),
                                dtype=tf.int32 if dataloader.features == 'phoneme' else tf.float32),
                    tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
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
    GLOBAL_BATCH_SIZE = 2048
    train_dataset = LibriPhraseDataloader(batch_size=GLOBAL_BATCH_SIZE, train=True, types='both', shuffle=True, pkl='/home/DB/LibriPhrase/data/train_both.pkl', features='g2p_embed')
    test_dataset = LibriPhraseDataloader(batch_size=GLOBAL_BATCH_SIZE, train=False, edit_dist=True, types='both', shuffle=False, pkl='/home/DB/LibriPhrase/data/test_both.pkl', features='g2p_embed')