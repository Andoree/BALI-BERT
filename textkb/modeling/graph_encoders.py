import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, DataParallel, TransformerConv, SimpleConv, SAGEConv
from torch.nn import functional as F


class GraphEncoderOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(input_tensor)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_channels, num_layers: int, num_hidden_channels, dropout_p: float, multigpu,
                 remove_output_dropout: bool = False, normalize: bool = False, project: bool = False):
        super().__init__()

        self.num_layers = num_layers
        self.num_hidden_channels = num_hidden_channels
        self.dropout_p = dropout_p
        self.remove_output_dropout = remove_output_dropout
        self.normalize = normalize
        self.project = project
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            input_num_channels = in_channels if i == 0 else num_hidden_channels

            output_num_channels = num_hidden_channels
            if i == num_layers - 1:
                output_num_channels = in_channels

            if multigpu:
                sage_conv = DataParallel(SAGEConv(input_num_channels, output_num_channels,
                                                  project=project, normalize=normalize))
            else:
                sage_conv = SAGEConv(input_num_channels, output_num_channels,
                                     project=project, normalize=normalize)

            self.convs.append(sage_conv)

        self.gelu = nn.GELU()
        if self.remove_output_dropout:
            self.out = GraphEncoderOutput(hidden_size=in_channels,
                                          hidden_dropout_prob=dropout_p)
        else:
            self.out = nn.Sequential(
                nn.Linear(in_channels, in_channels),
                nn.LayerNorm([in_channels, ], eps=1e-12, elementwise_affine=True),
                nn.Dropout(dropout_p)
            )

    def forward(self, x, edge_index, num_trg_nodes):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if not i == self.num_layers - 1:
                x = F.dropout(x, p=self.dropout_p, training=self.training)
                x = self.gelu(x)
        x = self.out(x[:num_trg_nodes])

        return x


class GATv2Encoder(nn.Module):
    def __init__(self, in_channels, num_layers: int, num_hidden_channels, dropout_p: float,
                 num_att_heads: int, attention_dropout_p: float, add_self_loops, multigpu,
                 remove_output_dropout: bool = False):
        super().__init__()
        self.num_layers = num_layers

        self.num_att_heads = num_att_heads
        self.num_hidden_channels = num_hidden_channels
        self.dropout_p = dropout_p
        self.remove_output_dropout = remove_output_dropout
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            input_num_channels = in_channels if i == 0 else num_hidden_channels

            output_num_channels = num_hidden_channels
            if i == num_layers - 1:
                output_num_channels = in_channels
            assert output_num_channels % num_att_heads == 0
            gat_head_output_size = output_num_channels // num_att_heads
            if multigpu:
                gat_conv = DataParallel(GATv2Conv(in_channels=input_num_channels, out_channels=gat_head_output_size,
                                                  heads=num_att_heads, dropout=attention_dropout_p,
                                                  add_self_loops=add_self_loops,
                                                  edge_dim=in_channels, share_weights=True))
            else:
                gat_conv = GATv2Conv(in_channels=input_num_channels, out_channels=gat_head_output_size,
                                     heads=num_att_heads, dropout=attention_dropout_p,
                                     add_self_loops=add_self_loops, edge_dim=in_channels, share_weights=True)
            self.convs.append(gat_conv)

        self.gelu = nn.GELU()
        if self.remove_output_dropout:
            self.out = GraphEncoderOutput(hidden_size=in_channels,
                                          hidden_dropout_prob=dropout_p)
        else:
            self.out = nn.Sequential(
                nn.Linear(in_channels, in_channels),
                nn.LayerNorm([in_channels, ], eps=1e-12, elementwise_affine=True),
                nn.Dropout(dropout_p)
            )

    def forward(self, x, edge_index, num_trg_nodes, return_attention_weights=False):
        for i, conv in enumerate(self.convs):
            if return_attention_weights:
                x, t = conv(x, edge_index, return_attention_weights=return_attention_weights)
            else:
                x = conv(x, edge_index, return_attention_weights=return_attention_weights)
            if not i == self.num_layers - 1:
                x = F.dropout(x, p=self.dropout_p, training=self.training)
                x = self.gelu(x)
        x = self.out(x[:num_trg_nodes])
        if return_attention_weights:
            return x, t
        else:
            return x


class TransformerConvLayer(nn.Module):
    def __init__(self, in_channels: int, heads: int, dropout_p: float):
        super().__init__()
        transformer_conv_out_channels = in_channels // heads
        self.conv = TransformerConv(in_channels=in_channels,
                                    out_channels=transformer_conv_out_channels,
                                    heads=heads,
                                    dropout=dropout_p)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)

        return x


class AttentionSumLayer(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.gat_conv = GATv2Conv(in_channels=in_channels,
                                  out_channels=in_channels,
                                  heads=1,
                                  dropout=0.,
                                  add_self_loops=False,
                                  edge_dim=None,
                                  share_weights=True)
        self.sum_conv = SimpleConv(aggr="sum")

    def forward(self, x, edge_index):
        _, (edge_index, alpha) = self.gat_conv(x, edge_index, return_attention_weights=True)
        x = self.sum_conv(x, edge_index, edge_weight=alpha)

        return x
