import math, os, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.io import wavfile
import torch

sys.path.append(os.path.dirname(__file__))
from g2p.g2p_en.g2p import G2p


class GoogleCommandsDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 batch_size,
                 fs = 16000,
                 wav_dir='/home/DB/google_speech_commands',
                 gemb_dir=None,
                 target_list=['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'],
                 features='g2p_embed', # phoneme, g2p_embed, both ...
                 shuffle=True,
                 testset_only=False,
                 pkl=None,
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
        self.target_list = [x.lower() for x in target_list]
        self.testset_only = testset_only
        self.features = features
        self.shuffle = shuffle
        self.pkl = pkl
        self.frame_length = frame_length
        self.hop_length = hop_length
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
                if self.testset_only:
                    test_list = os.path.join(self.wav_dir, 'testing_list.txt')
                    with open(test_list, "r") as f:
                        wav_list = f.readlines()
                        wav_list = [os.path.join(self.wav_dir, x.strip()) for x in wav_list]
                        wav_list = [x for x in wav_list if target == x.split('/')[-2]]
                else:
                    wav_list = [str(x) for x in Path(os.path.join(self.wav_dir, target)).rglob('*.wav')]
                for wav in wav_list:
                    anchor_text = wav.split('/')[-2].lower()
                    duration = float(wavfile.read(wav)[1].shape[-1])/self.fs
                    for comparison_text in self.target_list:
                        label = 1 if anchor_text == comparison_text else 0
                        target_dict[idx] = {
                            'wav': wav,
                            'text': comparison_text,
                            'duration': duration,
                            'label': label
                            }
                        idx += 1
            self.data = pd.concat([self.data, pd.DataFrame.from_dict(target_dict, 'index')], ignore_index=True)
    
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
        return self.len

    def _load_wav(self, wav):
        return np.array(wavfile.read(wav)[1]).astype(np.float32) / 32768.0
    
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
        
        if self.gemb_dir is not None:
            file_dirs = os.path.splitext(self.wav_list[i])[0]
            file_dirs = file_dirs.split("/")
            filepath = file_dirs[-1] + '.npy'
            dirpath = os.path.join(*file_dirs[3:-1])
            gemb = torch.from_numpy(np.load(os.path.join(self.gemb_dir, dirpath, filepath))[0]).to(torch.float32)
        else:
            gemb = None

        if self.features == 'both':
            return {"x": x, "gemb": gemb, "y": None, "p": p, "e": e, "z": z,}
        else:
            return {"x": x, "gemb": gemb, "y": y, "p": None, "e": None, "z": z,}
    
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
            batch = [{"x", "gemb", "y", "p", "e", "z",}]
        '''
        batch_dict = {
            "x": None,          "x_len": None,
            "gemb": None,       "gemb_len": None, 
            "y": None,          "y_len": None,
            "p": None,          "p_len": None,
            "e": None,          "e_len": None,
            "z": None,          "z_len": None,
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
        
        if self.gemb_dir is not None:
            batch_dict["gemb"] = self.pad_sequence([b["gemb"] for b in batch], int(int((self.maxlen_a - self.frame_length)/self.hop_length + 1)/8))
            batch_dict["gemb_len"] = torch.Tensor([int(int((b["x"].shape[0] - self.frame_length)/self.hop_length + 1)/8) for b in batch]).to(dtype=torch.int32, device=device)

        return batch_dict

