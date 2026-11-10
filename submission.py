import torch
from torch import nn
from torch_geometric.utils import add_self_loops, degree


class MyGCNConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        source, target = edge_index

        deg = degree(target, num_nodes, dtype=x.dtype).clamp(min=1)
        norm = deg[source].pow(-0.5) * deg[target].pow(-0.5)

        messages = self.lin(x)[source] * norm.unsqueeze(-1)
        out = x.new_zeros((num_nodes, self.lin.out_features))
        out.index_add_(0, target, messages)
        return out


class MySAGEConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_self = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        source, target = edge_index

        neigh_sum = torch.zeros_like(x)
        neigh_sum.index_add_(0, target, x[source])
        deg = degree(target, num_nodes, dtype=x.dtype).clamp(min=1)
        neigh_mean = neigh_sum / deg.unsqueeze(-1)

        return self.lin_neigh(neigh_mean) + self.lin_self(x)
