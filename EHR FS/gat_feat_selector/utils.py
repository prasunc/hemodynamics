from enum import Enum
import torch
import torch.nn.functional as F


class DistanceMetric(Enum):
    COSINE = "COSINE"


def calculate_distance(x1, x2=None, eps=1e-8, metric=DistanceMetric.COSINE):
    if metric == DistanceMetric.COSINE:
        x2 = x1 if x2 is None else x2
        w1 = torch.norm(x1, p=2, dim=1, keepdim=True)
        w2 = w1 if x2 is None else torch.norm(x2, p=2, dim=1, keepdim=True)
        return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

    raise Exception("This metric is not still implemented")


def find_sim_threshold(adj_mat, num_samples, edge_per_node):
    sorted_adj_mat = torch.sort(adj_mat.reshape(-1,), descending=True).values[edge_per_node * num_samples]
    sim_threshold = sorted_adj_mat.item()
    return sim_threshold


def get_adjacency_info(data, edge_per_node=10, self_loop=True):
    # print("\n data: ", data)

    adj_matrix = calculate_distance(data)
    # print("\n adj_matrix: ", adj_matrix)
    sim_threshold = find_sim_threshold(adj_matrix, data.shape[0], edge_per_node)
    # print("\n sim_threshold: ", sim_threshold)

    non_zero_entries = (adj_matrix >= sim_threshold).float()
    # print("\n non_zero_entries: ", non_zero_entries)
    if self_loop:
        non_zero_entries.fill_diagonal_(0)
    adj_matrix = torch.mul(adj_matrix, non_zero_entries)
    # print("\n new adj_matrix: ", adj_matrix)

    identity_mat = torch.eye(adj_matrix.shape[0], device=adj_matrix.device)
    # print("\n identity_mat: ", identity_mat)
    adj_matrix = F.normalize(adj_matrix + identity_mat, p=1)
    # print("\n Normalized adj_matrix with identity: ", adj_matrix)
    adj_matrix = adj_matrix.to_sparse()
    # print("\n Sparse adj_matrix: ", adj_matrix)

    return adj_matrix.indices(), adj_matrix.values()
