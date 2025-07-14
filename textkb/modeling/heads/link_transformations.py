import abc
import math
from abc import ABC

import torch
from torch import nn


class AbstractLinkTransformation(ABC, nn.Module):
    num_relations: int
    embedding_dim: int
    embedding_range: float

    def init_relation_matrices(self):
        self.register_parameter('w_relation', nn.Parameter(torch.Tensor(self.num_relations, self.embedding_dim)))
        self.embedding_range = 6. / math.sqrt(self.embedding_dim)
        with torch.no_grad():
            self.w_relation.uniform_(-self.embedding_range, self.embedding_range)

    @abc.abstractmethod
    def transform(self, head, rel_idx):
        raise NotImplementedError("head_transform")


class TransETransformation(AbstractLinkTransformation, nn.Module):
    def __init__(self, num_rels, h_dim, transe_margin_gamma=1.0, dist_ord=2):
        super().__init__()
        self.num_relations = num_rels
        self.embedding_dim = h_dim
        self.dist_ord = dist_ord
        self.transe_margin_gamma = transe_margin_gamma
        self.init_relation_matrices()

    def transform(self, head, rel_idx):
        relation = self.w_relation[rel_idx]
        tail = head + relation

        return tail


class DistMultTransformation(AbstractLinkTransformation, nn.Module):
    def __init__(self, num_rels, h_dim):
        super().__init__()
        self.num_relations = num_rels
        self.embedding_dim = h_dim
        self.init_relation_matrices()

    def transform(self, head, rel_idx):
        relation = self.w_relation[rel_idx]
        tail = head * relation

        return tail


class RotatETransformation(nn.Module):
    def __init__(self, num_rels, h_dim):
        super().__init__()
        self.num_relations = num_rels
        self.embedding_dim = h_dim
        self.init_relation_matrices()
        self.reset_parameters()
        self.score_matrix = None

    def init_relation_matrices(self):
        self.register_parameter('w_relation',
                                nn.Parameter(torch.Tensor(self.num_relations, self.embedding_dim)))
        self.add_module("node_emb_real", nn.Linear(self.embedding_dim, self.embedding_dim))
        self.add_module("node_emb_image", nn.Linear(self.embedding_dim, self.embedding_dim))
        # self.embedding_range = 6. / math.sqrt(self.embedding_dim)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.node_emb_real.weight)
        torch.nn.init.xavier_uniform_(self.node_emb_image.weight)
        torch.nn.init.uniform_(self.w_relation, 0, 2 * math.pi)

    def forward(self, head_emb, tail_emb, rel_idx):
        # <b, h> Splitting relation part
        rel_theta = self.w_relation[rel_idx]
        rel_re, rel_im = torch.cos(rel_theta), torch.sin(rel_theta)

        # <b, h> Mapping head embeddings to real and image parts
        head_re = self.node_emb_real(head_emb)
        head_im = self.node_emb_image(head_emb)
        b, h = head_re.size(0), head_re.size(1)

        # <b, h> Mapping tail embeddings to real and image parts
        tail_re = self.node_emb_real(tail_emb)
        tail_im = self.node_emb_image(tail_emb)

        # <b, 1, h>
        head_transformed_re = (rel_re * head_re - rel_im * head_im).unsqueeze(1)
        head_transformed_im = (rel_re * head_im + rel_im * head_re).unsqueeze(1)
        # <b, b, h>
        re_score = head_transformed_re - tail_re
        im_score = head_transformed_im - tail_im
        assert re_score.size() == (b, b, h)
        assert im_score.size() == (b, b, h)
        # <b, b, h, 2>
        complex_score = torch.stack([re_score, im_score], dim=-1)
        # <b, b>
        score = torch.linalg.vector_norm(complex_score, dim=(2, 3))

        self.score_matrix = score

        return score

    def get_score(self, *args, **kwargs):
        return self.score_matrix
