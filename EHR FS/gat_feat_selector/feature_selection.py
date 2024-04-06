import torch
import numpy as np
from torch_geometric.data import Data
import torch_geometric.transforms as T
from gat_feat_selector.utils import get_adjacency_info
from gat_feat_selector.model import GAT
from gat_feat_selector.train_test import train, test


class GATFeatSelector:
    def __init__(self, num_classes, num_epochs, edge_per_node=10, dropout_rate=0.5, hidden_channels=[64, 64, 64]):
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.edge_per_node = edge_per_node
        self.dropout_rate = dropout_rate
        self.hidden_channels = hidden_channels
        self.graph_data = None
        self.raw_data = None
        self.model = None
        self.total_acc = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def process(self, X, y):
        # print("X_train\n", X)
        # print("X_test\n", y)
        raw_data = torch.from_numpy(X).to(torch.float)
        labels = torch.from_numpy(y).to(torch.long)
        self.raw_data = raw_data
        # print("raw_data_tensor\n", self.raw_data)
        # print("labels_tensor\n", labels)

        edge_index, edge_attr = get_adjacency_info(self.raw_data, edge_per_node=self.edge_per_node)

        graph_data = Data(x=self.raw_data, edge_index=edge_index, edge_attr=edge_attr, y=labels)
        # print(graph_data)
        split = T.RandomNodeSplit(num_val=0.0, num_test=0.2)
        graph_data = split(graph_data)
        print(graph_data)

        self.graph_data = graph_data

        # print("\n\n\n")
        # print(graph_data)
        # print(graph_data.train_mask)
        # print(graph_data.test_mask)
        # print(graph_data.x)
        # print(graph_data.edge_index)
        # print(graph_data.edge_attr)

        # Gather some statistics about the graph.
        print(f'Number of nodes: {graph_data.num_nodes}')
        print(f'Number of edges: {graph_data.num_edges}')
        print(f'Average node degree: {graph_data.num_edges / graph_data.num_nodes:.2f}')
        print(f'Number of training nodes: {graph_data.train_mask.sum()}')
        print(f'Number of test nodes: {graph_data.test_mask.sum()}')
        print(f'Has isolated nodes: {graph_data.has_isolated_nodes()}')
        print(f'Has self-loops: {graph_data.has_self_loops()}')
        print(f'Is undirected: {graph_data.is_undirected()}')

    # def process(self, X_train, X_test, y_train, y_test):
    #     print("X_train\n", X_train)
    #     print("X_test\n", X_test)
    #     print("y_train\n", y_train)
    #     print("y_test\n", y_test)
    #     num_train = X_train.shape[0]
    #     num_test = X_test.shape[0]
    #     num_data_samples = num_train + num_test
    #     X_train_tensor = torch.from_numpy(X_train)
    #     X_test_tensor = torch.from_numpy(X_test)
    #     y_train_tensor = torch.from_numpy(y_train)
    #     y_test_tensor = torch.from_numpy(y_test)
    #     raw_data = torch.cat([X_train_tensor, X_test_tensor], dim=0).to(torch.float)
    #     labels = torch.cat([y_train_tensor, y_test_tensor], dim=0).to(torch.long)
    #     self.raw_data = raw_data
    #     print("raw_data_tensor\n", raw_data)
    #     print("labels_tensor\n", labels)
    #
    #     edge_index, edge_attr = get_adjacency_info(self.raw_data, edge_per_node=self.edge_per_node)
    #
    #     graph_data = Data(x=self.raw_data, edge_index=edge_index, edge_attr=edge_attr, y=labels)
    #     graph_data.train_mask = torch.zeros(num_data_samples, dtype=torch.bool)
    #     graph_data.train_mask[:num_train] = True
    #     graph_data.test_mask = torch.zeros(num_data_samples, dtype=torch.bool)
    #     graph_data.test_mask[num_train:] = True
    #
    #     self.graph_data = graph_data
    #
    #     print("\n\n\n")
    #     print(graph_data)
    #     print(graph_data.train_mask)
    #     print(graph_data.test_mask)
    #     print(graph_data.x)
    #     print(graph_data.edge_index)
    #     print(graph_data.edge_attr)
    #
    #     # Gather some statistics about the graph.
    #     print(f'Number of nodes: {graph_data.num_nodes}')
    #     print(f'Number of edges: {graph_data.num_edges}')
    #     print(f'Average node degree: {graph_data.num_edges / graph_data.num_nodes:.2f}')
    #     print(f'Number of training nodes: {graph_data.train_mask.sum()}')
    #     print(f'Number of test nodes: {graph_data.test_mask.sum()}')
    #     print(f'Has isolated nodes: {graph_data.has_isolated_nodes()}')
    #     print(f'Has self-loops: {graph_data.has_self_loops()}')
    #     print(f'Is undirected: {graph_data.is_undirected()}')

    def fit(self):
        self.model = GAT(in_channels=self.graph_data.num_features, hidden_channels=self.hidden_channels,
                         out_channels=self.num_classes, dropout=self.dropout_rate)
        print(self.model)

        self.model = self.model.to(self.device)
        self.graph_data = self.graph_data.to(self.device)
        print("device:", self.device)

        criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)  # Define optimizer.

        for epoch in range(1, self.num_epochs):
            loss = train(self.model, optimizer, criterion, self.graph_data)
            if epoch % 20 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

        acc = test(self.model, self.graph_data)
        print("Accuracy", acc)
        self.total_acc = acc

        return self._feature_importance()

    def _feature_importance(self):
        feat_importance = []
        for feat_idx in range(self.graph_data.num_features):
            # print("\n\n\n\n feature index:", feat_idx)
            # print("\n\n\n New test")
            temp_feat = self.raw_data[:, feat_idx].clone()
            self.raw_data[:, feat_idx] = 0
            # print("New data\n", self.raw_data)
            # print("temp_feat\n", temp_feat)
            edge_index, edge_attr = get_adjacency_info(self.raw_data, edge_per_node=self.edge_per_node)

            self.graph_data.x = self.raw_data
            self.graph_data.edge_index = edge_index
            self.graph_data.edge_attr = edge_attr
            # print(self.graph_data)
            # print(self.graph_data.x)
            # print(self.graph_data.edge_attr)

            self.graph_data = self.graph_data.to(self.device)
            # print("device:", self.device)

            acc_feat = test(self.model, self.graph_data)
            # print("Accuracy", acc_feat)
            feat_importance.insert(feat_idx, self.total_acc - acc_feat)

            self.raw_data[:, feat_idx] = temp_feat
            # print("Old data\n", self.raw_data)
            # print("temp_feat\n", temp_feat)

        # print(feat_importance)
        sorted_indices = sorted(range(len(feat_importance)), key=lambda k: feat_importance[k], reverse=True)
        print(sorted_indices)
        # sorted_values = [feat_importance[i] for i in sorted_indices]
        # print(sorted_values)

        return sorted_indices
