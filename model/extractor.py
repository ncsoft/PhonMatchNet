import os, sys
import torch
import torch.nn as nn
import numpy as np

sys.path.append(os.path.dirname(__file__))
from utils import make_adjacency_matrix, make_feature_matrix

seed = 42
torch.random.manual_seed(seed)
np.random.seed(seed)


class Extractor(nn.Module):
    """Base class for extractors"""
    
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, emb_s, emb_t, emb_s_mask=None, emb_t_mask=None):
        """
        Args:
            emb_s       : speech embedding of shape `(batch, time, embedding)`
            emb_t       : text embedding of shape `(batch, phoneme, embedding)`
            emb_s_mask  : mask indicating the lengths of speech embedding of shape `(batch, time)`
            emb_t_mask  : mask indicating the lengths of text embedding of shape `(batch, time)`
        """
        raise NotImplementedError


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding, num_heads):
        super().__init__()
        self.d_model = embedding
        self.num_heads = num_heads
        assert self.d_model % self.num_heads == 0, "d_model % num_heads should be zero."

        self.d_head = int(self.d_model / self.num_heads)
        self.sqrt_dim = torch.math.sqrt(self.d_model)

        self.query_proj = nn.Linear(self.d_model, self.d_model)
        self.key_proj = nn.Linear(self.d_model, self.d_model)
        self.value_proj = nn.Linear(self.d_model, self.d_model)

        self.out_proj = nn.Linear(self.d_model, self.d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = value.shape[0]

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head) 
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        """
        shapes
            query      : `(Batch, L_q, Heads, Feature/Heads)`
            key, value : `(Batch, Heads, L_k, Feature/Heads)`
        """

        attention_score = torch.matmul(query.transpose(1, 2), key.transpose(2, 3)) # Q x K^T, `(Batch, Heads, L_q, L_k)`
        attention_score = attention_score / self.sqrt_dim

        if mask is not None:
            attn_mask = torch.logical_not(mask.unsqueeze(1))
            attention_score = attention_score.masked_fill_(attn_mask, -np.inf)
        
        attention_prob = torch.nn.functional.softmax(attention_score, dim=-1)   # `(Batch, Heads, L_q, L_k)`
        out = torch.matmul(torch.nan_to_num(attention_prob), value)             # `(Batch, Heads, L_q, L_k)` @ `(Batch, Heads, L_k, Feature/Heads)`
        out = out.transpose(1, 2)                                               # `(Batch, L_q, Heads, Feature/Heads)`
        out = out.contiguous().view(batch_size, -1, self.d_model)               # `(Batch, L_q, Feature)`
        if mask is not None:
            out = torch.nan_to_num(out) * mask[:,:,0].unsqueeze(-1)

        return self.out_proj(out), attention_score
    

class BaseExtractor(Extractor):
    """Base class for pattern extractor"""
    
    def __init__(self, **kwargs):
        super().__init__()
        self.embedding = kwargs['embedding']
        self.num_heads = kwargs['num_heads']
        self.attn = MultiHeadAttention(self.embedding, self.num_heads)
            
    def forward(self, emb_s, emb_t, emb_s_mask=None, emb_t_mask=None, verbose=False):
        """
        Args:
            emb_s       : speech embedding of shape `(batch, time, embedding)`
            emb_t       : text embedding of shape `(batch, phoneme, embedding)`
            emb_s_mask  : mask indicating the lengths of speech embedding of shape `(batch, time)`
            emb_t_mask  : mask indicating the lengths of text embedding of shape `(batch, time)`
            * Query - text, Key,Value - speech *
        """
        Q = emb_t
        V = emb_s
        
        # [B, Tt, m], [B, Tt, Ta] notation followed Learning Audio-Text Agreement for Open-vocabulary Keyword Spotting
        if (emb_s_mask is not None) and (emb_t_mask is not None):
            if emb_t_mask is None:
                attn_mask = None
            else:
                attn_mask = emb_t_mask.unsqueeze(-1) * emb_s_mask.unsqueeze(1)
        else:
            attn_mask = None
        affn_mask = None

        attention_output, affinity_matrix = self.attn(Q, V, V,
                                                mask = attn_mask,
                                                )
        
        if self.num_heads == 1:
            affinity_matrix = affinity_matrix[:,0,:,:]

        if attn_mask is not None:
            affn_mask = attn_mask
            attn_mask = attn_mask[:,:,0]
            affinity_matrix = torch.nan_to_num(affinity_matrix) * affn_mask
            attention_output = torch.nan_to_num(attention_output) * attn_mask.unsqueeze(-1)

        return attention_output, affinity_matrix, attn_mask, affn_mask
        

class StackExtractor(Extractor):
    """Self-attention based pattern extractor"""
    
    def __init__(self, **kwargs):
        super().__init__()
        self.embedding = kwargs['embedding']
        self.num_heads = kwargs['num_heads']
        self.attn = MultiHeadAttention(self.embedding, self.num_heads)
            
    def forward(self, emb_s, emb_t, emb_s_mask=None, emb_t_mask=None, verbose=False):
        """
        Args:
            emb_s   : speech embedding of shape `(batch, time, embedding)`
            emb_t   : text embedding of shape `(batch, phoneme, embedding)`
            * Query - text, Key,Value - speech *
        """
        Q = make_feature_matrix(emb_s, emb_t, emb_s_mask, emb_t_mask)
        V = Q
        attn_mask = make_adjacency_matrix(emb_s_mask, emb_t_mask)
        
        attention_output, affinity_matrix = self.attn(Q, V, V,
                                                mask = attn_mask,
                                                )
        
        if self.num_heads == 1:
            affinity_matrix = affinity_matrix[:,0,:,:]
        if attn_mask is not None:
            affn_mask = attn_mask
            attn_mask = attn_mask[:,:,0]
            affinity_matrix = torch.nan_to_num(affinity_matrix) * affn_mask
            attention_output = torch.nan_to_num(attention_output) * attn_mask.unsqueeze(-1)
        else:
            affn_mask = None
            
        return attention_output, affinity_matrix, attn_mask, affn_mask