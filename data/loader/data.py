import torch
from torch import Tensor
import torch_geometric
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import degree
from torch_geometric.datasets import Planetoid, TUDataset
from .utils import pool_edge
from torch_scatter import scatter
import torch_cluster
from typing import List, Tuple
import copy
from torch_geometric.graphgym.config import cfg

import data
from .coarsener import HierarchyGenerator, RandomHierarchyGenerator
import os
from tqdm import tqdm
import time

"""
This file contains the implementation of the HierarchicalGraphData class.
Both classes are adapted from:

Hierarchical Transformer for Scalable Graph Learning
https://arxiv.org/abs/2305.02866
"""


class HierarchyFeatureCollator:
    """
    This class collates node features for the hierarchical
    support graphs.
    """
    def __init__(
            self,
            collator_type: str):
        """
        Args:
            collator_type (str): collator type ('mean' or 'mode')
        """
        self.collator_type = collator_type

    def __call__(
            self,
            data: torch_geometric.data.Data,
            hier_map: List[Tensor]
    ) -> List[Tensor]:
        collated_feature = [data.x]
        if self.collator_type == "mean":
            for i in range(len(hier_map)):
                collated_feature.append(scatter(
                    src=collated_feature[i],
                    index=hier_map[i],
                    dim=0,
                    reduce="mean"
                ))
        elif self.collator_type == "mode":
            for i in range(len(hier_map)):
                collated_feature.append(scatter(
                    src=collated_feature[i],
                    index=hier_map[i],
                    dim=0,
                    reduce="mode"
                ))
        else:
            raise NotImplementedError("Feature Collator undefined.")
        return collated_feature


class HierarchicalGraphData:
    """
    This class is a wrapper for the creation of hierarchy
    of support graphs. It also collates node features and
    pools edges for the higher level graphs.
    """
    def __init__(
            self,
            graph_data: torch_geometric.data.Data,
            hierarchy_generator: HierarchyGenerator,
            feature_collator: HierarchyFeatureCollator,
    ):
        """
        Creates hierarchical graph data, allows for method interchangeability.

        Args:
            graph_data (torch_geometric.data.Data): original graph data
            hierarchy_generator (HierarchyGenerator): hierarchy generator
            feature_collator (HierarchyFeatureCollator): feature collator
        """
        self.org_data = graph_data
        self.y = graph_data.y
        self.num_leaf_nodes: int = graph_data.num_nodes
        self.hier_map = hierarchy_generator(graph_data)  # List[Tensor]
        self.num_hierarchies = len(self.hier_map) + 1
        self.collated_feature = feature_collator(graph_data, self.hier_map)

        self.hier_data = [self.org_data]
        self.degrees = [degree(
            index=self.org_data.edge_index[0, :],
            num_nodes=self.num_leaf_nodes
        ).int()]

        # remove duplicates
        coarsened_edge_index = self.org_data.edge_index.contiguous()
        if self.org_data.edge_attr is not None:
            coarsened_edge_attr = self.org_data.edge_attr.contiguous()
        else:
            coarsened_edge_attr = None

        # recursive coarsening loop
        for i in range(len(self.hier_map)):
            num_nodes = self.collated_feature[i + 1].shape[0]

            coarsened_edge_index, coarsened_edge_attr = pool_edge(
                cluster=self.hier_map[i],
                edge_index=coarsened_edge_index,
                edge_attr=coarsened_edge_attr,
                reduce=cfg.hsg.edge_reduce_type, 
            )

            coarsened_data = Data(
                x=self.collated_feature[i + 1],
                edge_index=coarsened_edge_index,
                edge_attr=coarsened_edge_attr,
                num_nodes=num_nodes
            )

            self.hier_data.append(coarsened_data)
            # calculate node degrees
            coarsened_data_degree = degree(
                index=coarsened_data.edge_index[0, :],
                num_nodes=num_nodes
            ).int()

            if coarsened_data_degree.numel() == 0:
                coarsened_data_degree = torch.tensor([0], dtype=torch.int)
            self.degrees.append(coarsened_data_degree)

        for idx in range(len(self.hier_data)):
            self.hier_data[idx] = self.hier_data[idx].coalesce()

    def __len__(self):
        return self.num_hierarchies

    def __getitem__(self, ind: int):
        return self.hier_data[ind]