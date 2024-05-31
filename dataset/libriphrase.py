import math, os, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import Levenshtein
from multiprocessing import Pool
from scipy.io import wavfile
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(__file__))
from g2p.g2p_en.g2p import G2p


class LibriPhraseDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 batch_size,
                 fs = 16000,
                 wav_dir='/home/DB/LibriPhrase_diffspk_all',
                 gemb_dir=None,
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
                 frame_length=None,
                 hop_length=None,
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
        self.gemb_dir = gemb_dir
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
        self.frame_length = frame_length
        self.hop_length = hop_length
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
                    self.data = pd.concat([self.data, anc_pos.rename(columns={y: x for x, y in zip(self.data.columns, anc_pos.columns)})], ignore_index=True)
                    self.data = pd.concat([self.data, anc_neg.rename(columns={y: x for x, y in zip(self.data.columns, anc_neg.columns)})], ignore_index=True)
                    self.data = pd.concat([self.data, com_pos.rename(columns={y: x for x, y in zip(self.data.columns, com_pos.columns)})], ignore_index=True)
                    self.data = pd.concat([self.data, com_neg.rename(columns={y: x for x, y in zip(self.data.columns, com_neg.columns)})], ignore_index=True)

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
        return self.len

    def _load_wav(self, wav):
        return np.array(wavfile.read(wav)[1]).astype(np.float32) / 32768.0
    
    def _mixing_snr(self, clean, snr=[5, 15]):
        def _cal_adjusted_rms(clean_rms, snr):
            a = float(snr) / 20
            noise_rms = clean_rms / (10**a) 
            return noise_rms

        def _cal_rms(amp):
            return torch.sqrt(torch.mean(torch.square(amp), axis=-1))
        
        start = torch.randint(0, len(self.noise)-len(clean), size=(1,))
        divided_noise = torch.Tensor(self.noise[start: start + len(clean)]).to(torch.float32)
        clean_rms = _cal_rms(clean)
        noise_rms = _cal_rms(divided_noise)
        adj_noise_rms = _cal_adjusted_rms(clean_rms, np.random.randint(snr[0], snr[1]))
        
        adj_noise_amp = divided_noise * (adj_noise_rms / (noise_rms + 1e-7)) 
        noisy = clean + adj_noise_amp
        
        if torch.max(noisy) > 1:
            noisy = noisy / torch.max(noisy)
        
        return noisy
    
    def __getitem__(self, idx):
        # chunking
        i = self.indices[idx]
        
        # load inputs
        x = torch.Tensor(wavfile.read(self.wav_list[i])[1]).to(torch.float32) / 32768.0
        if self.features == 'both':
            p = torch.Tensor(self.idx_list[i]).to(torch.int32)
            e = torch.Tensor(self.emb_list[i]).to(torch.float32)
        else:
            if self.features == 'phoneme':
                y = torch.Tensor(self.idx_list[i]).to(torch.int32)
            elif self.features == 'g2p_embed':
                y = torch.Tensor(self.emb_list[i]).to(torch.float32)
        
        # load outputs
        z = torch.Tensor([self.lab_list[i]]).to(torch.float32)
        l = torch.Tensor(self.sIdx_list[i]).to(torch.int32)
        t = torch.Tensor(self.idx_list[i]).to(torch.int32)
        if self.edit_dist:
            d = torch.Tensor([self.dist_list[i]]).to(torch.float32)

        # Noisy option
        if self.train:
            x_noisy = self._mixing_snr(x)
        
        if self.gemb_dir is not None:
            file_dirs = os.path.splitext(self.wav_list[i])[0]
            file_dirs = file_dirs.split("/")
            filepath = file_dirs[-1] + '.npy'
            dirpath = os.path.join(*file_dirs[3:-1])
            gemb = torch.from_numpy(np.load(os.path.join(self.gemb_dir, dirpath, filepath))[0]).to(torch.float32)
        else:
            gemb = None
            
        if self.train:
            if self.features == 'both':
                return {"x": x, "x_noisy": x_noisy, "gemb": gemb, "y": None, "p": p, "e": e, "z": z, "l": l, "t": t, "d": None,}
            else:
                return {"x": x, "x_noisy": x_noisy, "gemb": gemb, "y": y, "p": None, "e": None, "z": z, "l": l, "t": t, "d": None,}
        else:
            if self.features == 'both':
                if self.edit_dist:
                    return {"x": x, "x_noisy": None, "gemb": gemb, "y": None, "p": p, "e": e, "z": z, "l": None, "t": None, "d": d,}
                else:
                    return {"x": x, "x_noisy": None, "gemb": gemb, "y": None, "p": p, "e": e, "z": z, "l": None, "t": None, "d": None,}
            else:
                if self.edit_dist:
                    return {"x": x, "x_noisy": None, "gemb": gemb, "y": y, "p": None, "e": None, "z": z, "l": None, "t": None, "d": d,}
                else:
                    return {"x": x, "x_noisy": None, "gemb": gemb, "y": y, "p": None, "e": None, "z": z, "l": None, "t": None, "d": None,}

    def on_epoch_end(self):
        self.indices = np.arange(self.len)
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def pad_sequence(self, data, max_len):
        pad_list = [0 for _ in range(data[0].dim()*2)]
        pad_list[-1] = max_len - data[0].shape[0]
        data[0] = torch.nn.functional.pad(data[0], tuple(pad_list))
        return torch.nn.utils.rnn.pad_sequence(data, batch_first=True)

    def collate(self, batch):
        '''
            batch = [{"x", "x_noisy", "gemb", "y", "p", "e", "z", "l", "t", "d",}]
        '''
        batch_dict = {
            "x": None,          "x_len": None,
            "x_noisy": None,    "x_noisy_len": None,
            "gemb": None,       "gemb_len": None, 
            "y": None,          "y_len": None,
            "p": None,          "p_len": None,
            "e": None,          "e_len": None,
            "z": None,          "z_len": None,
            "l": None,          "l_len": None,
            "t": None,          "t_len": None,
            "d": None,          "d_len": None,
            }
        
        device = batch[0]["x"].device
        batch_dict["x"] = self.pad_sequence([b["x"] for b in batch], self.maxlen_a)
        batch_dict["z"] = torch.nn.utils.rnn.pad_sequence([b["z"] for b in batch], batch_first=True)
        batch_dict["x_len"] = torch.Tensor([b["x"].shape[0] for b in batch]).to(dtype=torch.int32, device=device)
        # batch_dict["z_len"] = torch.Tensor([b["z"].shape[0] for b in batch]).to(torch.int32) # always [1,1, ...]
        
        if self.features == 'both':
            batch_dict["p"] = self.pad_sequence([b["p"] for b in batch], self.maxlen_t)
            batch_dict["e"] = self.pad_sequence([b["e"] for b in batch], self.maxlen_t)
            batch_dict["p_len"] = torch.Tensor([b["p"].shape[0] for b in batch]).to(dtype=torch.int32, device=device)
            batch_dict["e_len"] = torch.Tensor([b["e"].shape[0] for b in batch]).to(dtype=torch.int32, device=device)
        else:
            batch_dict["y"] = self.pad_sequence([b["y"] for b in batch], self.maxlen_t)
            batch_dict["y_len"] = torch.Tensor([b["y"].shape[0] for b in batch]).to(dtype=torch.int32, device=device)
        
        if self.train:
            batch_dict["x_noisy"] = self.pad_sequence([b["x_noisy"] for b in batch], self.maxlen_a)
            batch_dict["l"] = self.pad_sequence([b["l"] for b in batch], self.maxlen_l)
            batch_dict["t"] = self.pad_sequence([b["t"] for b in batch], self.maxlen_t)
            # batch_dict["x_noisy_len"] = torch.Tensor([b["x_noisy"].shape[0] for b in batch]).to(torch.int32) # identical to x_len
            batch_dict["l_len"] = torch.Tensor([b["l"].shape[0] for b in batch]).to(dtype=torch.int32, device=device)
            batch_dict["t_len"] = torch.Tensor([b["t"].shape[0] for b in batch]).to(dtype=torch.int32, device=device)
        
        if self.gemb_dir is not None:
            batch_dict["gemb"] = self.pad_sequence([b["gemb"] for b in batch], int(int((self.maxlen_a - self.frame_length)/self.hop_length + 1)/8))
            batch_dict["gemb_len"] = torch.Tensor([int(int((b["x"].shape[0] - self.frame_length)/self.hop_length + 1)/8) for b in batch]).to(dtype=torch.int32, device=device)
        
        elif self.edit_dist:
            batch_dict["d"] = torch.nn.utils.rnn.pad_sequence([b["d"] for b in batch], batch_first=True)
            batch_dict["d_len"] = torch.Tensor([b["d"].shape[0] for b in batch]).to(dtype=torch.int32, device=device)

        return batch_dict