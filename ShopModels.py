# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 23:48:14 2019

@author: User
"""
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class Pair_Shop_MLP(torch.nn.Module):
    def __init__(self, config):
        super(Pair_Shop_MLP, self).__init__()
        self.fc_layers = torch.nn.ModuleList()
        # 这个是层的变量
        for idx, (in_size, out_size) in enumerate(zip(config['shop_mlp_layers'][:-1], config['shop_mlp_layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

    def forward(self, shops_embedding, distances_embedding, ratings, comments, prices_embedding, group):
        ratings = ratings.view(ratings.shape[0], ratings.shape[1], 1)
        comments = comments.view(comments.shape[0], comments.shape[1], 1)
        group = group.view(group.shape[0], group.shape[1], 1)
        vector = torch.cat([shops_embedding, distances_embedding,
                           ratings, comments, prices_embedding, group], dim=-1)
        for idx in range(len(self.fc_layers)):
            vector = self.fc_layers[idx](vector)
            vector = torch.nn.ReLU()(vector)
        return vector


class GCN1Layers(torch.nn.Module):
    def __init__(self, config):
        super(GCN1Layers, self).__init__()
        input_size = config['GCNLayers'][0]
        # hidden_size = config['GCNLayers'][1]
        output_size = config['GCNLayers'][2]
        self.conv1 = GCNConv(input_size, output_size)
        # x = F.relu(x)

        # self.conv2 = GCNConv(hidden_size, output_size)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x = self.conv2(x, edge_index)

        return x


class GCN_Shop_MLP(torch.nn.Module):
    def __init__(self, config):
        super(GCN_Shop_MLP, self).__init__()
        self.GNN_embedding_types = torch.nn.Embedding(config['n_types']+1,
                                                      config['type_hidden_size'],
                                                      padding_idx=config['n_types'])

        self.embedding_GNN_prices = torch.nn.Embedding(config['n_prices']+1,
                                                       config['price_hidden_size'],
                                                       padding_idx=config['n_prices'])

    def forward(self, all_types, all_prices, all_groups):
        GNN_type_embedding = self.GNN_embedding_types(all_types.long())
        GNN_prices_embedding = self.embedding_GNN_prices(all_prices.long())

        return torch.cat([GNN_type_embedding, GNN_prices_embedding, all_groups.view(all_groups.shape[0], -1)], dim=-1)


class GCN2Layers(torch.nn.Module):
    def __init__(self, config):
        super(GCN2Layers, self).__init__()
        input_size = config['GCNLayers'][0]
        hidden_size = config['GCNLayers'][1]
        output_size = config['GCNLayers'][2]
        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, output_size)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x
