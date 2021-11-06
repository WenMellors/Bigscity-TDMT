import torch.nn as nn
import torch.nn.functional as F

from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


class LINE_FIRST(nn.Module):
    def __init__(self, num_nodes, embedding_size):
        super().__init__()
        self.num_nodes = num_nodes
        self.embedding_size = embedding_size
        self.node_emb = nn.Embedding(self.num_nodes, self.embedding_size)

    def forward(self, i, j):
        """
        Args:
            i: indices of i; (B,)
            j: indices of j; (B,)
        Return:
            v_i^T * v_j; (B,)
        """
        vi = self.node_emb(i)
        vj = self.node_emb(j)
        return (vi * vj).sum(dim=-1)


class LINE_SECOND(nn.Module):
    def __init__(self, num_nodes, embedding_size):
        super().__init__()
        self.num_nodes = num_nodes
        self.embedding_size = embedding_size
        self.node_emb = nn.Embedding(self.num_nodes, self.embedding_size)
        self.context_emb = nn.Embedding(self.num_nodes, self.embedding_size)

    def forward(self, I, J):
        """
        Args:
            I: indices of i; (B,)
            J: indices of j; (B,)
        Return:
            [v_i^T * u_j for (i,j) in zip(I,J)]; (B,)
        """
        vi = self.node_emb(I)
        vj = self.context_emb(J)
        return (vi * vj).sum(dim=-1)


class LINE(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.device = config.get('device')

        self.order = config.get('order')
        self.embedding_size = config.get('embedding_size')
        self.num_nodes = data_feature.get("num_nodes")
        self.num_edges = data_feature.get("num_edges")

        if self.order == 'first':
            self.embed = LINE_FIRST(self.num_nodes, self.embedding_size)
        elif self.order == 'second':
            self.embed = LINE_SECOND(self.num_nodes, self.embedding_size)
        else:
            raise ValueError("order mode must be first or second")

    def calculate_loss(self, batch):
        I, J, is_neg = batch['I'], batch['J'], batch['Neg']
        dot_product = self.forward(I, J)
        return -(F.logsigmoid(dot_product * is_neg)).mean()

    def forward(self, I, J):
        """
        Args:
            I : origin indices of node i ; (B,)
            J : origin indices of node j ; (B,)
        Return:
            if order == 'first':
                [u_j^T * u_i for (i,j) in zip(I, J)]; (B,)
            elif order == 'second':
                [u'_j^T * v_i for (i,j) in zip(I, J)]; (B,)
        """
        return self.embed(I, J)
