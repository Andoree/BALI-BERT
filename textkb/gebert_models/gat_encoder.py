from torch import nn as nn
from torch.nn import functional as F
from torch_geometric.nn import GATv2Conv


class GeBertGATv2Encoder(nn.Module):
    def __init__(self, in_channels, num_outer_layers: int, num_inner_layers: int, num_hidden_channels, dropout_p: float,
                 num_att_heads: int, attention_dropout_p: float,
                 add_self_loops, layernorm_output):
        super().__init__()
        self.num_outer_layers = num_outer_layers

        self.num_inner_layers = num_inner_layers
        self.num_att_heads = num_att_heads
        self.num_hidden_channels = num_hidden_channels
        self.dropout_p = dropout_p
        self.convs = nn.ModuleList()
        for i in range(num_outer_layers):
            inner_convs = nn.ModuleList()
            for j in range(num_inner_layers):
                input_num_channels = in_channels if (j == 0 and i == 0) else num_hidden_channels

                output_num_channels = num_hidden_channels
                if (i == num_outer_layers - 1) and (j == num_inner_layers - 1):
                    output_num_channels = in_channels
                assert output_num_channels % num_att_heads == 0
                gat_head_output_size = output_num_channels // num_att_heads
                gat_conv = GATv2Conv(in_channels=input_num_channels, out_channels=gat_head_output_size,
                                     heads=num_att_heads, dropout=attention_dropout_p,
                                     add_self_loops=add_self_loops, edge_dim=in_channels, share_weights=True)
                inner_convs.append(gat_conv)

            self.convs.append(inner_convs)
        self.gelu = nn.GELU()
        self.lin_proj = nn.Linear(in_channels, in_channels)
        self.layernorm_output = layernorm_output
        if self.layernorm_output:
            self.out = nn.Sequential(
                nn.Linear(in_channels, in_channels),
                nn.LayerNorm([in_channels, ], eps=1e-12, elementwise_affine=True),
                nn.Dropout(dropout_p)
            )

    def forward(self, x, edge_index, num_trg_nodes):
        # x, edge_index, num_trg_nodes
        for i, inner_convs_list in enumerate(self.convs):
            for j, conv in enumerate(inner_convs_list):
                x = conv(x, edge_index=edge_index)
                if not (i == self.num_outer_layers - 1 and j == self.num_inner_layers - 1):
                    x = F.dropout(x, p=self.dropout_p, training=self.training)
                    x = self.gelu(x)
        if self.layernorm_output:
            x = self.out(x[:num_trg_nodes])
        return x
