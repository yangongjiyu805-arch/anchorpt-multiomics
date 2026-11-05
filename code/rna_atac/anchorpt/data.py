"""Prepare multi-omics observations for PyTorch data loading.

The dataset preserves sparse inputs until individual rows are requested. It also
packages covariates, modality indicators, feature-domain indicators, labels, and
optional Stage-I anchor representations for the training pipeline.
"""

import numpy as np
import torch
from scipy.sparse import issparse
from torch.utils.data import Dataset


class OmicsDataset(Dataset):
    def __init__(self, view_data, batch_cov, omics_id, domain_id, label_cov, anchor=None):
        super().__init__()
        self._view_is_sparse = issparse(view_data)
        if self._view_is_sparse:
            self.view_data = view_data.tocsr().astype(np.float32, copy=False)
        else:
            self.view_data = torch.tensor(view_data).float()
        self.len = view_data.shape[0]
        self.batch_cov = torch.tensor(batch_cov).float() if batch_cov is not None else torch.zeros((self.len, 0)).float()
        self.omics_id = torch.tensor(omics_id).float()
        self.domain_id = torch.tensor(domain_id).float()
        self.label_cov = torch.tensor(label_cov).float() if label_cov is not None else torch.zeros((self.len, 0)).float()
        self.anchor = torch.tensor(anchor).float() if anchor is not None else torch.zeros((self.len, 0)).float()

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self._view_is_sparse:
            view_row = torch.from_numpy(self.view_data[index].toarray().ravel()).float()
        else:
            view_row = self.view_data[index]
        return view_row, self.batch_cov[index], self.omics_id[index], self.domain_id[index], self.label_cov[index], self.anchor[index]
