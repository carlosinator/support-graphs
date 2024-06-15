import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import new_layer_config, MLP
from torch_geometric.graphgym.register import register_head


@register_head('inductive_masked_node')
class GNNInductiveMaskedNodeHead(nn.Module):
    """
    GNN prediction head for inductive node prediction tasks.
    Based on the original 'inductive_node' head in head/inductive_node.py.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """

    def __init__(self, dim_in, dim_out):
        super(GNNInductiveMaskedNodeHead, self).__init__()
        self.layer_post_mp = MLP(
            new_layer_config(dim_in, dim_out, cfg.gnn.layers_post_mp,
                             has_act=False, has_bias=True, cfg=cfg))

    def _apply_index(self, batch):
        if 'real_mask' not in batch:
            raise ValueError("real_mask not found in batch")

        x = batch.x
        y = batch.y

        mask = batch['real_mask'] # only output values for nodes in the lowest hierarchy layer
        return x[mask], y[mask]

    def forward(self, batch):
        batch = self.layer_post_mp(batch)
        pred, label = self._apply_index(batch)
        return pred, label
