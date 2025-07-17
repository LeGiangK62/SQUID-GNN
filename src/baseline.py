import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GCNConv, GATConv, SAGEConv, TransformerConv, MLP, global_add_pool

class GIN_Node(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            mlp = MLP([in_channels if i==0 else hidden_channels,
                       hidden_channels, hidden_channels])
            self.convs.append(GINConv(nn=mlp, train_eps=False))
        self.dropout = nn.Dropout(0.8)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_attr, edge_index, batch=None):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = self.dropout(x)
        return x
    


class GCN_Node(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_ch  = in_channels if i==0 else hidden_channels
            out_ch = out_channels  if i==num_layers-1 else hidden_channels
            self.convs.append(GCNConv(in_ch, out_ch))
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_attr, edge_index, batch=None):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = self.dropout(x)
        # last layer, no activation
        return x


class GAT_Node(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers, heads=8, dropout=0.6):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i==0 else hidden_channels * heads
            if i < num_layers-1:
                self.convs.append(
                    GATConv(in_ch, hidden_channels, heads=heads, dropout=dropout)
                )
            else:
                # final layer: single head, no concat
                self.convs.append(
                    GATConv(in_ch, out_channels, heads=1, concat=False, dropout=dropout)
                )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_attr, edge_index, batch=None):
        for conv in self.convs:
            x = F.elu(conv(x, edge_index))
            x = self.dropout(x)
        return x
    
    
## NOTE: Graph Task

class GIN_Graph(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            mlp = MLP([in_channels if i==0 else hidden_channels,
                       hidden_channels, hidden_channels])
            self.convs.append(GINConv(nn=mlp, train_eps=False))
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_attr, edge_index, batch=None):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = self.dropout(x)
        x = global_add_pool(x, batch)
        return self.classifier(x)
    
class GCN_Graph(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            self.convs.append(GCNConv(in_dim, hidden_channels))
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_attr, edge_index, batch=None):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = self.dropout(x)
        x = global_add_pool(x, batch)
        return self.classifier(x)
    
class GAT_Graph(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads=1):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels * heads
            self.convs.append(GATConv(in_dim, hidden_channels, heads=heads))
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_channels * heads, out_channels)

    def forward(self, x, edge_attr, edge_index, batch=None):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = self.dropout(x)
        x = global_add_pool(x, batch)
        return self.classifier(x)


class GraphSAGE_Graph(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            self.convs.append(SAGEConv(in_dim, hidden_channels))
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_attr, edge_index, batch=None):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = self.dropout(x)
        x = global_add_pool(x, batch)
        return self.classifier(x)
    
    
class Transformer_Graph(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads=1):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels * heads
            self.convs.append(TransformerConv(in_dim, hidden_channels, heads=heads))
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_channels * heads, out_channels)

    def forward(self, x, edge_attr, edge_index, batch=None):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = self.dropout(x)
        x = global_add_pool(x, batch)
        return self.classifier(x)