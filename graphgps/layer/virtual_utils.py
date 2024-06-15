import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.graphgym import cfg
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.models.pooling import global_mean_pool
import torch


class NodeOneHot(nn.Module):
    """
    pre-processing layer that adds one-hot embeddings to NODE features.

    Args:
    dim_out: the dimension of the embedding
    """
    def __init__(self, dim_out):
        super().__init__()
        self.dim_in = 1 + len(cfg.hsg.num_hierarchy_nodes)
        self.dim_out = dim_out

        self.model = nn.Embedding(self.dim_in, self.dim_out)

    def forward(self, batch):
        if not hasattr(batch, "node_onehot"):
            raise ValueError("batch needs 'node_onehot' attribute.")

        if cfg.hsg.mask_hsg_nodes: 
            # replace higher level node feats with dummies
            mask_fake = batch.node_onehot > 0
            batch.x = torch.where(
                mask_fake[:,None], self.model(batch.node_onehot), batch.x)
        else:
            batch.x = batch.x + self.model(batch.node_onehot)

        return batch


class EdgeOneHot(nn.Module):
    """
    pre-processing layer that adds one-hot embeddings to EDGE features.

    Args:
    dim_out: the dimension of the embedding
    """
    def __init__(self, dim_out):
        super().__init__()
        # all hierarchies (len+1) + the "vertical edges"
        self.dim_in =  1 + 2 * len(cfg.hsg.num_hierarchy_nodes) 
        self.dim_out = dim_out

        self.model = nn.Embedding(self.dim_in, self.dim_out)

    def forward(self, batch):
        if not hasattr(batch, "edge_onehot"):
            raise ValueError("batch needs 'edge_onehot' attribute.")

        type_embs = self.model(batch.edge_onehot.type(torch.long))

        if cfg.hsg.mask_hsg_edges:
            # replace higher level edge feats with dummies
            mask_fake = batch.edge_onehot > 0 
            batch.edge_attr = torch.where(
                mask_fake[:,None], type_embs, batch.edge_attr) 
        else:
            batch.edge_attr = type_embs + batch.edge_attr

        return batch

