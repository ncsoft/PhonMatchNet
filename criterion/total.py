import os, sys
import torch
import torch.nn as nn
import numpy as np
# from tensorflow.keras.losses import Loss, MeanSquaredError

seed = 42
torch.random.manual_seed(seed)
np.random.seed(seed)

def sequence_cross_entropy(speech_label, text_label, logits, logits_mask, reduction='sum'):
    """
    args
    speech_label        : [B, Ls]
    text_label          : [B, Lt]
    logits              : [B, Lt]
    logits_mask         : [B, Lt]
    """
    # Data pre-processing
    if text_label.shape[1] > speech_label.shape[1]:
        speech_label = nn.functional.pad(speech_label, (0, text_label.shape[1] - speech_label.shape[1]), 'constant', value=0)
    elif text_label.shape[1] < speech_label.shape[1]:
        speech_label = speech_label[:, :text_label.shape[1]]
    
    # Make paired data between text and speech phonemes
    paired_label = torch.logical_and(text_label == speech_label, logits_mask).float()
    paired_label = torch.masked_select(paired_label, logits_mask.bool()).view(-1, 1)
    logits = torch.masked_select(logits, logits_mask.bool()).view(-1, 1)
    
    # Get BinaryCrossEntropy loss
    BCE = nn.BCEWithLogitsLoss(reduction=reduction)
    loss = BCE(logits.float(), paired_label.float())
    
    if reduction == 'sum':
        loss = torch.nan_to_num(loss)
        loss = loss / logits.shape[0]
        loss = torch.nan_to_num(loss)
        loss = loss * speech_label.shape[0]
        loss = torch.nan_to_num(loss)

    return loss

def detection_loss(y_true, y_pred, reduction):
    BCE = nn.BCEWithLogitsLoss(reduction=reduction)
    return(BCE(y_pred.float(), y_true.float()))


class TotalLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
    
    def forward(self, y_true, y_pred, reduction='sum'):
        LD = detection_loss(y_true, y_pred, reduction)

        return self.weight * LD, LD


class TotalLoss_SCE(nn.Module):
    def __init__(self, weight=[1.0, 1.0]):
        super().__init__()
        self.weight = weight
    
    def forward(self, y_true, y_pred, speech_label, text_label, logits, logits_mask, reduction='sum'):
        if self.weight[0] != 0.0:   
            LD = detection_loss(y_true, y_pred, reduction)
        else:
            LD = 0
        if self.weight[1] != 0.0:
            LC = sequence_cross_entropy(speech_label, text_label, logits, logits_mask, reduction=reduction)
        else:
            LC = 0
        return self.weight[0] * LD + self.weight[1] * LC, LD, LC


