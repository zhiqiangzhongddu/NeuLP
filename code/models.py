import torch
import torch.nn as nn
import torch_geometric as tg
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

class GNN(torch.nn.Module):
    def __init__(self, config, in_dim):
        super(GNN, self).__init__()
        self.config = config
        self.gnn = config['gnn']
        self.drop_ratio = config['drop_ratio']
        self.in_dim = in_dim

        self.gnn_layers = torch.nn.ModuleList()
        if self.gnn == 'GCN':
            if self.config['feature_pre']:
                self.linear_pre = torch.nn.Linear(self.in_dim, self.config['gnn_layers'][0])
            for idx, (in_size, out_size) in enumerate(
                    zip(self.config['gnn_layers'][:-1], self.config['gnn_layers'][1:])):
                self.gnn_layers.append(GCNConv(in_size, out_size))
        elif self.gnn == 'SAGE':
            if self.config['feature_pre']:
                self.linear_pre = torch.nn.Linear(self.in_dim, self.config['gnn_layers'][0])
            for idx, (in_size, out_size) in enumerate(
                    zip(self.config['gnn_layers'][:-1], self.config['gnn_layers'][1:])):
                self.gnn_layers.append(SAGEConv(in_size, out_size))
        elif self.gnn == 'GAT':
            if self.config['feature_pre']:
                self.linear_pre = torch.nn.Linear(self.in_dim, self.config['gnn_layers'][0])
            for idx, (in_size, out_size) in enumerate(
                    zip(self.config['gnn_layers'][:-1], self.config['gnn_layers'][1:])):
                self.gnn_layers.append(GATConv(in_size, out_size))
    def gnn_embedding(self, data):
        x, edge_index = data.x, data.edge_index

        # x = torch.nn.BatchNorm1d(num_features=x.shape[1]).to(device)(x)
        if self.config['feature_pre']:
            x = self.linear_pre(x)
        embed = x
        for idx, _ in enumerate(range(len(self.gnn_layers))):
            if idx != len(self.gnn_layers) - 1:
                embed = self.gnn_layers[idx](embed, edge_index)
                embed = torch.nn.functional.relu(embed)
                embed = torch.nn.functional.dropout(embed, p=self.drop_ratio, training=self.training)
            else:
                embed = self.gnn_layers[idx](embed, edge_index)
        embed = torch.nn.functional.normalize(embed, p=2, dim=-1)
        return embed


class NeuLP(torch.nn.Module):
    def __init__(self, config, in_dim):
        super(NeuLP, self).__init__()
        self.config = config
        self.crit = torch.nn.BCELoss()

        self.gnn = GNN(self.config, in_dim)

        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(
                zip(self.config['mlp_layers'][:-1], self.config['mlp_layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(
            in_features=self.gnn.gnn_layers[-1].out_channels + self.config['mlp_layers'][-1], out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, data, user_indices_left, user_indices_right):
        gnn_emb = self.gnn.gnn_embedding(data)
        left_users_embedding = gnn_emb[user_indices_left]
        right_users_embedding = gnn_emb[user_indices_right]

        mf_vector = torch.mul(left_users_embedding, right_users_embedding)
        mlp_vector = torch.cat([left_users_embedding, right_users_embedding], dim=-1)

        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)
            mlp_vector = torch.nn.ReLU()(mlp_vector)

        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

