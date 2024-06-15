from typing import Optional

import torch

from torch_geometric.utils import coalesce, remove_self_loops, scatter, sort_edge_index


def pool_edge(
    cluster,
    edge_index,
    edge_attr: Optional[torch.Tensor] = None,
    reduce: Optional[str] = 'sum'):
    """
    This is a pool_edge function from a higher PyG version that
    allows for different edge reduce operations.
    Args:
        cluster (torch.Tensor): cluster tensor
        edge_index (torch.Tensor): edge index tensor
        edge_attr (torch.Tensor): edge attribute tensor
        reduce (str): reduce operation
    Returns:
        torch.Tensor: reduced edge index tensor
        torch.Tensor: reduced edge attribute tensor
    """
    ea_dtype = edge_attr.dtype if edge_attr is not None else None

    num_nodes = cluster.size(0)
    edge_index = cluster[edge_index.view(-1)].view(2, -1)
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    if edge_index.numel() > 0:
        if reduce != "mode":
            edge_index, edge_attr = coalesce(
                edge_index, edge_attr, 
                num_nodes,reduce=reduce)
        else:
            edge_index, edge_attr = reduce_mode(
                edge_index, edge_attr, num_nodes)
    if edge_attr is not None:
        edge_attr = edge_attr.type(ea_dtype)
    return edge_index, edge_attr


def reduce_mode(edge_index, edge_attr, num_nodes):
    """
    This function implements a mode reduction function for edge_attr,
    based on the PyG implementation for other reduce operations.

    Args:
        edge_index (torch.Tensor): edge index tensor
        edge_attr (torch.Tensor): edge attribute tensor
        num_nodes (int): number of nodes in the graph
    Returns:
        torch.Tensor: reduced edge index tensor
        torch.Tensor: reduced edge attribute tensor
    """
    new_edge_index, _ = coalesce(edge_index, None, num_nodes, reduce='mean')
    edge_index, edge_attr = sort_edge_index(edge_index, edge_attr)
    num_edges = edge_index.size(1)
    idx = edge_index[0].new_empty(num_edges + 1)
    idx[0] = -1
    idx[1:] = edge_index[0]
    idx[1:].mul_(num_nodes).add_(edge_index[1])

    mask = idx[1:] > idx[:-1]

    # Only perform expensive merging in case there exists duplicates:
    if mask.all():
        if edge_attr is None or isinstance(edge_attr, (torch.Tensor, list, tuple)):
            return edge_index, edge_attr
        return edge_index
    
    dim_size: Optional[int] = None
    if isinstance(edge_attr, (torch.Tensor, list, tuple)) and len(edge_attr) > 0:
        dim_size = edge_index.size(1)
        idx = torch.arange(0, num_edges, device=edge_index.device)
        idx.sub_(mask.logical_not_().cumsum(dim=0))

    # expensive for loop
    ea_size = edge_attr.size(1)
    new_edge_attr = torch.zeros(
        (new_edge_index.size(1), ea_size), dtype=edge_attr.dtype)

    for i in range(idx.max()+1):
        new_edge_attr[i] = edge_attr[idx == i].mode(dim=0)[0]

    return new_edge_index, new_edge_attr