import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=3, dropout=0.5):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels[0], heads=num_heads, concat=False, edge_dim=1)
        self.conv2 = GATConv(hidden_channels[0], hidden_channels[1], heads=num_heads, concat=False, edge_dim=1)
        self.conv3 = GATConv(hidden_channels[1], hidden_channels[2], heads=num_heads, concat=False, edge_dim=1)
        self.classifier = Linear(hidden_channels[2], out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index, edge_attr)
        x = F.leaky_relu(x, 0.25)

        x = self.classifier(x)

        return x
