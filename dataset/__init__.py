import torch
from torch.utils.data import Dataset, DataLoader


class KWSDataLoader(DataLoader):
    def __init__(
            self,
            dataset: Dataset,
            batch_size: int = 1,
            shuffle: bool = False,
            follow_batch=None,
            **kwargs,
    ):
        if follow_batch is None:
            follow_batch = []
        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        self.follow_batch = follow_batch

        super().__init__(dataset, batch_size, shuffle, collate_fn=dataset.collate, **kwargs)
