import torch
import torch.nn.functional as F

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.init import init_weights
from torch_geometric.graphgym.models.layer import (
    BatchNorm1dNode,
    GeneralLayer,
    GeneralMultiLayer,
    new_layer_config,
)
from torch_geometric.graphgym.register import register_stage

from graphgps.layer.virtual_utils import NodeOneHot, EdgeOneHot



class ExtendedFeatureEncoder(torch.nn.Module):
    r"""Encodes node and edge features, given the specified input dimension and
    the underlying configuration in :obj:`cfg`.

    Adapted from the original `FeatureEncoder` in `/network/gps_model.py`.

    Args:
        dim_in (int): The input feature dimension.
    """
    def __init__(self, dim_in: int):
        super().__init__()
        self.dim_in = dim_in
        if cfg.dataset.node_encoder:
            # Encode integer node features via `torch.nn.Embedding`:
            NodeEncoder = register.node_encoder_dict[
                cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(
                        cfg.gnn.dim_inner,
                        -1,
                        -1,
                        has_act=False,
                        has_bias=False,
                        cfg=cfg,
                    ))
            # Update `dim_in` to reflect the new dimension fo the node features
            self.dim_in = cfg.gnn.dim_inner
        if cfg.dataset.edge_encoder:
            # Encode integer edge features via `torch.nn.Embedding`:
            EdgeEncoder = register.edge_encoder_dict[
                cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.gnn.dim_inner)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(
                        cfg.gnn.dim_inner,
                        -1,
                        -1,
                        has_act=False,
                        has_bias=False,
                        cfg=cfg,
                    ))

        # add onehot embedding for hierarchical nodes and edges
        if cfg.hsg.use_node_onehot:
            self.onehot_emb = NodeOneHot(self.dim_in)
        if cfg.hsg.use_edge_onehot:
            self.edge_onehot = EdgeOneHot(self.dim_in)



    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch