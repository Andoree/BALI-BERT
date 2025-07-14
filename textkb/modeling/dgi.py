from abc import ABC

import torch
from torch import Tensor

EPS = 1e-15


class AbstractDGIModel(ABC):
    @staticmethod
    def summary_fn(z, *args, **kwargs):
        batch_size = kwargs.get("batch_size")
        if batch_size is not None:
            z = z[:batch_size]
        return torch.sigmoid(z.mean(dim=0))

    @staticmethod
    def corruption_fn(embs, edge_index, *args, **kwargs):
        edge_index_src = edge_index[0]
        edge_index_trg = edge_index[1]
        num_edges = len(edge_index_trg)
        perm_trg_nodes = torch.randperm(num_edges)
        corr_edge_index_trg = edge_index_trg[perm_trg_nodes]
        corr_edge_index = torch.stack((edge_index_src, corr_edge_index_trg)).to(edge_index.device)

        return embs, corr_edge_index

    def discriminate(self, z, summary, sigmoid=True):
        r"""Given the patch-summary pair :obj:`z` and :obj:`summary`, computes
        the probability scores assigned to this patch-summary pair.

        This method is the modified version of the implementation from PyTorch Geometric but it calculates 'value'
        variable using torch.float32 datatype instead of torch.float16. Without this fix the model may fail to compute
        DGI loss.

        Args:
            z (Tensor): The latent space.
            summary (Tensor): The summary vector.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        summary = summary.t() if summary.dim() > 1 else summary
        value = torch.matmul(z, torch.matmul(self.dgi_weight, summary)).float()
        return torch.sigmoid(value) if sigmoid else value

    def dgi_loss(self, pos_z: Tensor, neg_z: Tensor, summary: Tensor) -> Tensor:
        r"""Computes the mutual information maximization objective."""
        pos_loss = -torch.log(
            self.discriminate(pos_z, summary, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(1 - self.discriminate(neg_z, summary, sigmoid=True) + EPS).mean()

        return pos_loss + neg_loss
