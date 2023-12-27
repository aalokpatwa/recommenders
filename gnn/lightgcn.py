# LightGCN
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree
import torch

from constants import OUTPUT_EMBEDDING

class Layer(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x, edge_list):
        """ Each layer simply computes a weighted sum of each node's neighbors' embeddings

        Args:
            x (torch.FloatTensor): N*D tensor containing embeddings for all N nodes
            edge_list (torch.LongTensor): list of edges in the graph

        Returns:
            N*D tensor 
        """
        # Normalize the aggregation by the degree of the node and its neighbor
        from_, to_ = edge_list
        deg = degree(to_, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]

        # Update the embeddings across the graph
        return self.propagate(edge_list, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

class LightGCN(nn.Module):
    def __init__(self, latent_dim, num_layers, num_users, num_items):
        super(LightGCN, self).__init__()

        # Only parameters: initial Embedding Layer
        self.embedding = nn.Embedding(num_users + num_items, latent_dim)

        # Define each LightGCN layer
        self.convs = nn.ModuleList(Layer() for _ in range(num_layers))

        self.init_parameters()

    def init_parameters(self):
        # Weight initialization
        nn.init.normal_(self.embedding.weight, std=0.1) 

    def forward(self, edge_list):
        """ LightGCN forward pass

        Args:
            edge_list (torch.LongTensor): edge list for the graph

        Returns:
            torch.FloatTensor: N * D output embeddings
        """
        emb0 = self.embedding.weight
        embs = [emb0]

        emb = emb0

        # The initial embeddings (from parameters) get updated through the graph convolution operation
        for conv in self.convs:
            emb = conv(x=emb, edge_index=edge_list)
            embs.append(emb)

        # In the paper, the output is the mean of the embeddings at each layer
        # We can also try just outputting the ending embedidng
        stacked = torch.stack(embs, dim=0)
        end_embedding = embs[-1]
        mean_embedding = torch.mean(stacked, dim=0)

        out = mean_embedding if OUTPUT_EMBEDDING=="mean" else end_embedding
        
        return emb0, out

    def encode_minibatch(self, users, pos_items, neg_items, edge_index):
        """ Helper to separate out the embeddings for users, positive items, negative items """
        emb0, out = self(edge_index)
        return (
            out[users], 
            out[pos_items], 
            out[neg_items], 
            emb0[users],
            emb0[pos_items],
            emb0[neg_items]
        )