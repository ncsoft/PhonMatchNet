import numpy as np
import sklearn.metrics
import torch
from torchmetrics import Metric

def compute_eer(label, pred):
    # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
    label = torch.nan_to_num(label).detach().cpu().numpy()
    pred = torch.nan_to_num(pred).detach().cpu().numpy()
    fpr, tpr, threshold = sklearn.metrics.roc_curve(label, pred)
    fnr = 1 - tpr

    # the threshold of fnr == fpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2
    return eer

class eer(Metric):
    # reference: https://pypi.org/project/torchmetrics/
    def __init__(self, **kwargs):
        super().__init__()
        self.add_state("score", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")

    def update(self, y_true, y_pred):
        self.score += torch.tensor(compute_eer(y_true, y_pred), dtype=self.score.dtype)
        self.count += torch.tensor(1.0)

    def compute(self):
        if self.count == 0:
            return torch.tensor(0, dtype=torch.float)
        score = torch.nan_to_num(self.score).float()
        return score / self.count