from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('cfg_hsg')
def set_cfg_gt(cfg):
    cfg.hsg = CN()
    cfg.hsg.use_node_onehot = False
    cfg.hsg.use_edge_onehot = False
    cfg.hsg.edge_reduce_type = "mean"
    cfg.hsg.mask_hsg_edges = False
    cfg.hsg.mask_hsg_nodes = False
    cfg.hsg.coarsen_method = "metis"
    cfg.hsg.num_hierarchy_nodes = []
    return