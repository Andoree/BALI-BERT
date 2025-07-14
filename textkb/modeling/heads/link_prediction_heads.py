import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinkScorer(nn.Module):
    def __init__(self, num_rels, h_dim, link_regularizer_weight, score_type, link_negative_adversarial_sampling=True,
                 link_negative_adversarial_sampling_temperature=1.0, transe_margin_gamma=1.0, dist_ord=2):
        super().__init__()
        self.num_relations = num_rels
        self.embedding_dim = h_dim
        self.score_type = score_type
        assert score_type in ("transe", "distmult")
        if self.score_type == "transe":
            self.dist_ord = dist_ord
            self.transe_margin_gamma = transe_margin_gamma
        self.init_relation_matrices(score_type)
        # self.epsilon = epsilon
        self.negative_adversarial_sampling = link_negative_adversarial_sampling
        self.adversarial_temperature = link_negative_adversarial_sampling_temperature
        self.link_regularizer_weight = link_regularizer_weight

        self.scaled_distmult = False

    def forward(self, head_embs, tail_embs, rel_idx, mode='single'):
        """
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head' or 'tail' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
        """

        relation = self.w_relation[rel_idx].unsqueeze(1)
        if head_embs.dim() < 3:
            head_embs = head_embs.unsqueeze(1)
        if tail_embs.dim() < 3:
            tail_embs = tail_embs.unsqueeze(1)

        score = self.score(head_embs, relation, tail_embs, mode)  # [n_triple, 1 or n_neg]

        return score

    def transe_score(self, head, relation, tail, mode):
        """
        Input head/tail has stdev 1 for each element. Scale to stdev 1/sqrt(12) * (b-a) = a/sqrt(3).
        Reference: https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/wikikg2/model.py
        """

        if mode == 'head':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.transe_margin_gamma - torch.norm(score, p=self.dist_ord, dim=2)
        return score

    def distmult_score(self, head, relation, tail, mode):
        if mode == 'head':
            if self.scaled_distmult:
                tail = tail / math.sqrt(self.embedding_dim)
            score = head * (relation * tail)
        else:
            if self.scaled_distmult:
                head = head / math.sqrt(self.embedding_dim)

            score = (head * relation) * tail

        score = score.sum(dim=2)
        return score

    def init_relation_matrices(self, model_type):
        self.register_parameter('w_relation', nn.Parameter(torch.Tensor(self.num_relations, self.embedding_dim)))
        self.embedding_range = 6. / math.sqrt(self.embedding_dim)
        with torch.no_grad():
            self.w_relation.uniform_(-self.embedding_range, self.embedding_range)
        # if model_type == "transe":
        #     self.register_parameter('w_relation', nn.Parameter(torch.Tensor(self.num_relations, self.embedding_dim)))
        #     # self.embedding_range = (self.transe_margin_gamma + self.epsilon) / self.embedding_dim
        #     self.embedding_range = 6. / math.sqrt(self.hidden_channels)
        #     with torch.no_grad():
        #         self.w_relation.uniform_(-self.embedding_range, self.embedding_range)
        # elif model_type == "distmult":
        #     logging.info("Initializing w_relation for DistMultDecoder...")
        #     self.register_parameter('w_relation', nn.Parameter(torch.Tensor(self.num_relations, self.embedding_dim)))
        #     self.embedding_range = math.sqrt(1.0 / self.embedding_dim)
        #     with torch.no_grad():
        #         self.w_relation.uniform_(-self.embedding_range, self.embedding_range)
        # else:
        #     raise NotImplementedError(f"Link scoring type: {model_type} is not implemented")

    def score(self, h, r, t, mode):
        if self.score_type == "transe":
            return self.transe_score(h, r, t, mode)
        elif self.score_type == "distmult":
            return self.distmult_score(h, r, t, mode)
        else:
            raise NotImplementedError(f"Unsupported link scoring function: {self.score_type}")

    def reg_loss(self):
        return torch.mean(self.w_relation.pow(2))
        # return torch.tensor(0)

    def loss(self, scores):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        positive_score, negative_score = scores
        if self.negative_adversarial_sampling:
            # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * self.adversarial_temperature, dim=1).detach()
                              * F.logsigmoid(-negative_score)).sum(dim=1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)  # [n_triple,]

        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)  # [n_triple,]

        assert positive_score.dim() == 1
        if len(positive_score) == 0:
            positive_sample_loss = negative_sample_loss = 0.
        else:
            # TODO: why???
            positive_sample_loss = - positive_score.mean()  # scalar
            negative_sample_loss = - negative_score.mean()  # scalar

        loss = (positive_sample_loss + negative_sample_loss) / 2 + self.link_regularizer_weight * self.reg_loss()

        return loss, positive_sample_loss, negative_sample_loss

