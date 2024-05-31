import math
import torch
import torch.nn as nn


def sequence_mask(length, max_length=None):
  if max_length is None:
    max_length = length.max()
  x = torch.arange(max_length, dtype=length.dtype, device=length.device)
  return x.unsqueeze(0) < length.unsqueeze(1)


def make_adjacency_matrix(speech_mask, text_mask):
    """
    args
    speech_mask : [B, Ls]
    text_mask   : [B, Lt]
    """
    # [B, L] -> [B]
    n_speech = torch.sum(speech_mask, -1)
    n_text = torch.sum(text_mask, -1)
    
    n_node = n_speech + n_text
    max_len = torch.max(n_node)

    # [B] -> [B, max_len] -> [B, max_len, 1] * [B, 1, max_len]-> [B, max_len, max_len]
    mask = sequence_mask(n_node, max_length=max_len)
    mask = mask.unsqueeze(-1) * mask.unsqueeze(1)
    # Make upper triangle matrix for adj. matrix
    adjacency_matrix = torch.tril(mask)
    
    return adjacency_matrix

def make_feature_matrix(speech_features, text_features, speech_mask, text_mask):
    """
    args
    speech_features : [B, Ls, F]
    speech_mask     : [B, Ls]
    text_features   : [B, Lt, F]
    text_mask       : [B, Lt]
    """
    # Concatenate two feature matrix along time axis
    feature_matrix = torch.cat((speech_features, text_features), dim=1) 
    feature_mask = torch.cat((speech_mask, text_mask), dim=1)
    
    n_speech = torch.sum(speech_mask, -1)
    n_text = torch.sum(text_mask, -1)
    
    n_node = n_speech + n_text
    max_len = torch.max(n_node)
    
    # Gather valid data using mask
    # [Warning] This returns a ragged tensor
    # Results are different with "feature_matrix * feature_mask.unsqueeze(-1)"
    indices = torch.masked_fill(torch.cumsum(feature_mask.int(), dim=1), ~feature_mask, 0)
    masked = torch.zeros(feature_matrix.shape[0], feature_matrix.shape[1]+1, feature_matrix.shape[2]).to(feature_matrix.device)
    masked = torch.scatter(input=masked, dim=1, index=torch.stack([indices for _ in range(feature_matrix.shape[-1])], dim=-1), src=feature_matrix)
    masked = masked[:,1:max_len+1]

    return masked