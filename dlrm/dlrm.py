import torch
from torch import nn
from torch.nn import functional as F

from constants import *

class BottomMLP(nn.Module):
    def __init__(self, n_inputs, output_dim, hidden_dim, n_layers):
        super().__init__()
        self.fc1 = nn.Linear(n_inputs, hidden_dim)

        layerlist = []

        for layer in range(n_layers - 2):
            layerlist.append(nn.Linear(hidden_dim, hidden_dim))
            layerlist.append(nn.ReLU())
        
        self.hidden = nn.Sequential(*layerlist)

        self.output = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # X is of shape (B * N_Inputs)
        x = F.relu(self.fc1(x))
        x = self.hidden(x)
        x = F.sigmoid(self.output(x))

        return x
    
class TopMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        layerlist = []
        
        for layer in range(n_layers - 2):
            layerlist.append(nn.Linear(hidden_dim, hidden_dim))
            layerlist.append(nn.ReLU())

        self.hidden = nn.Sequential(*layerlist) 

        # TODO: change back to 5 for categorical predictions
        self.output = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # X is of shape (B * N_Inputs)
        x = F.relu(self.fc1(x))
        x = self.hidden(x)
        x = self.output(x)

        return x

class FeatureInteraction(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # X is a list of shape (C * batch * Embedding_Dim)
        C = len(x)
        batch_size = x[0].shape[0]
        output_dim = int(C * (C+1) / 2)

        features = torch.stack(x, dim=1) # (batch * c * embedding_dim)

        dotted = features.matmul(torch.transpose(features, 1, 2)) # (batch * c * c)
        ones = torch.ones_like(dotted, dtype=torch.bool)
        mask = torch.triu(ones)

        # Flatten interactions from the top triangle of the interaction matrid
        flattened = torch.masked_select(dotted, mask).reshape((batch_size, output_dim))

        return flattened


class DLRM(nn.Module):
    def __init__(self, n_continuous, n_categorical, levels_per):
        super().__init__()
        print (f"Initializing DLRM model with n_continuous={n_continuous} and n_categorical={n_categorical}")
        self.n_continuous = n_continuous
        self.n_categorical = n_categorical
        self.bottom_mlp = BottomMLP(n_continuous, EMBEDDING_DIM, BOTTOM_MLP_HIDDEN_DIM, BOTTOM_MLP_LAYERS)
        self.embeddings = nn.ModuleList([nn.Embedding(levels_per[c], EMBEDDING_DIM) for c in range(n_categorical)])
        self.interaction = FeatureInteraction()

        top_mlp_dim = int(EMBEDDING_DIM + ((n_categorical + 1) * (n_categorical + 2) / 2))
        self.top_mlp = TopMLP(top_mlp_dim, TOP_MLP_HIDDEN_DIM, TOP_MLP_LAYERS)

    def forward(self, continuous, categorical):
        """ Forward pass of the DLRM

        Args:
            continuous (torch.FloatTensor): B * n_continuous tensor of continuous features
            categorical (torch.LongTensor): B * n_categorical tensor of categorical features

        Returns:
            torch.floatTensor: B * 1 tensor of predicted rankings
        """
        # Both continuous and categorical are B*num_features
        each_categorical = torch.unbind(categorical, dim=1) # (Categorical-length list of B*1 tensors)

        # Apply the bottom MLP, resulting in a B * D tensor
        continuous_embedding = self.bottom_mlp(continuous)

        # Categorical embeddings will be a list of length n_categorical, where each element is (B*D)
        categorical_embeddings = []

        for i in range(self.n_categorical):
            this_feature = each_categorical[i]
            embedding_layer = self.embeddings[i]
            this_embedding = embedding_layer(this_feature)
            categorical_embeddings.append(this_embedding)

        # List of length (n_categorical + 1), each element is B * D
        all_embeddings = categorical_embeddings + [continuous_embedding]

        # Get second-order interactions, resulting in B * (C+1 * C+2 / 2) tensor
        interactions = self.interaction(all_embeddings)

        # Concatenate second-order interactions with continuous embedding
        final_embedding = torch.cat((continuous_embedding, interactions), dim=1)

        # Apply top MLP to get (B*1) predictions
        output = self.top_mlp(final_embedding)

        return output