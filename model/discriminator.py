import torch
import torch.nn as nn
import numpy as np

seed = 42
torch.random.manual_seed(seed)
np.random.seed(seed)


class Discriminator(nn.Module):
    """Base class for discriminators"""
    
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, src, src_len=None):
        """
        Args:
            src      : source of shape `(batch, src_len)`
            src_mask : mask indicating the lengths of each source of shape `(batch, time)`
        """
        raise NotImplementedError


class BaseDiscriminator(Discriminator):
    """Base class for discriminators"""
    
    def __init__(self, **kwargs):
        super().__init__()
        self.gru = []
        input_size = kwargs['input_dim']
        for i, l in enumerate(kwargs['gru']):
            unit = l
            self.gru.append(nn.GRU(input_size, unit[0], batch_first=True))
            input_size = unit[0]
                
        self.gru = nn.ModuleList(self.gru)
        self.dense = nn.Linear(kwargs['gru'][-1][0], 1)
        self.act = nn.Sigmoid()

    def forward(self, src, src_mask=None, verbose=False):
        """
        Args:
            src         : source of shape `(batch, time, feature)`
            src_mask    : mask indicating the lengths of each source of shape `(batch, time)`
            verbose     : prints hidden vectors' shape
        """
        x = src
        hidden = None
        for layer in self.gru:
            if src_mask is not None:
                x = torch.nan_to_num(x) * src_mask.unsqueeze(-1)
            x, hidden = layer(x, hidden) # (B, T, embedding)
        
        n_src = torch.sum(src_mask, -1) - torch.tensor(1.0)
        x = x[torch.arange(n_src.shape[0]).to(n_src.device).long(), n_src.long()]      # Take only final features (B, embedding)
        
        x = self.dense(x)                # (B, 1)

        return self.act(x), x
        
