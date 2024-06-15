import copy
import time
import os

import torch
import torch_geometric.transforms
from torch import Tensor
import torch_geometric
from torch_geometric.nn import graclus, knn_graph
from torch_geometric.nn.pool.pool import pool_edge
import torch_geometric.transforms as T
import torch_sparse
from torch_sparse import SparseTensor
from typing import List, Tuple, Optional
from torch_geometric.graphgym.config import cfg
import torch_geometric.graphgym.register as register
from sklearn.cluster import KMeans
from torch_scatter import scatter
from torch_geometric.data import Data

VERBOSE = False

"""
define an uninitialized node encoder to be used for clustering
the encoder has random weights but is identical for all graphs
for increased stability.
"""
enc_name = cfg.dataset.node_encoder_name
if "+" in enc_name:
    enc_name = enc_name.split("+")[0]
ENCODER = register.node_encoder_dict[enc_name](cfg.gnn.dim_inner)



def rearrange_cluster_map(cluster_map: Tensor) -> Tensor:
    """function to remove empty clusters and reindex cluster_map"""

    vals = cluster_map.unique().sort()[0].tolist()
    vals_map = {val: i for i, val in enumerate(vals)}
    # cluster_map_list = list(cluster_map)
    rearranged_map = list(map(lambda x: vals_map[int(x)], cluster_map))
    return torch.tensor(rearranged_map)


class HierarchyGenerator:
    """
    Wrapper for different hierarchy generators
    """
    def __init__(
            self,
            generator_type: str,
            num_nodes: List[int],
            dataset_name: Optional[str] = None):
        """
        Args:
        generator_type: str, type of generator, random, metis or kmeans
        num_nodes: List[int], number of nodes in each layer 
            (0 -> original number of noder, 
             1: -> coarsening strength in each round)
        dataset_name: Optional[str], name of dataset, used for saving partition
        """
        self.generator_type = generator_type
        self.dataset_name = dataset_name
        if generator_type == "random":
            self.coarsener = RandomHierarchyGenerator(num_nodes)
        elif generator_type == "metis":
            self.coarsener = MetisHierarchyGenerator(num_nodes)
        elif generator_type == "kmeans":
            self.coarsener = KMeansHierarchyGenerator(num_nodes)
        else:
            raise NotImplementedError(
                f"Hierarchy Generator {generator_type} undefined.")

    def __call__(self, data: torch_geometric.data.Data):
        return self.coarsener(data, self.dataset_name)


class RandomHierarchyGenerator:
    """
    Random hierarchy generator as a baseline.

    Hierarchical Transformer for Scalable Graph Learning
    https://arxiv.org/abs/2305.02866
    """
    def __init__(self, num_nodes: List[int]):
        self.num_layers = len(num_nodes)
        self.num_nodes = num_nodes

    def __call__(
        self, 
        data: torch_geometric.data.Data, 
        dataset_name: Optional[str]) -> List[Tensor]:
        """ creates random hierarchy maps """
        hier_map = []
        sizes = [data.num_nodes]
        for i in range(self.num_layers):
            cluster_map = rearrange_cluster_map(torch.randint(0, self.num_nodes[i], (sizes[i],)))
            hier_map.append(cluster_map)
            sizes.append(cluster_map.max() + 1)
        return hier_map


class KMeansHierarchyGenerator:
    """
    KMeans hierarchy generator
    Provides clustering not based on connectivity but based
    on the similarity of node embeddings.
    """
    def __init__(self, num_nodes: List[int]):
        self.num_layers = len(num_nodes)
        self.num_nodes = num_nodes

    
    def __call__(self, data: torch_geometric.data.Data, dataset_name: Optional[str] = None) -> List[Tensor]:
        """
        Uses sklearn to perform KMeans clustering on random node embeddings.
        Particularily useful for datasets with integer node features.

        Args:
            data: torch_geometric.data.Data, input graph
            dataset_name: Optional[str], name of dataset, used for saving partition
        Returns:
            hier_map: List[Tensor], list of cluster maps for each layer
        """
        save_path = None
        if dataset_name is not None:
            folder_path = f"./partitions/kmeans_{dataset_name[0]}/{dataset_name[1]}"
            save_path = f"{folder_path}/{dataset_name[2]}_{self.num_nodes}.pt"
            if os.path.exists(save_path):
                hier_map = torch.load(save_path)
                if VERBOSE:
                    print(f"Loaded saved partition file {save_path}.")
                return hier_map
            
        hier_map = []
        sizes = [data.num_nodes]
        node_vecs = ENCODER(Data(x=copy.copy(data.x))).x.detach().cpu().numpy()
        for i in range(self.num_layers):
            if self.num_nodes[i] >  node_vecs.shape[0]:
                self.num_nodes[i] = node_vecs.shape[0]
            km = KMeans(n_clusters=self.num_nodes[i]).fit(node_vecs)
            cluster_map = torch.tensor(km.labels_, dtype=torch.int64)
            hier_map.append(cluster_map)

            node_vecs = scatter(
                    src=torch.tensor(node_vecs),
                    index=cluster_map,
                    dim=0,
                    reduce="mean"
                ).detach().numpy()

        if save_path is not None and self.num_layers > 0:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            torch.save(hier_map, save_path)

        return hier_map


class MetisHierarchyGenerator:
    """
    Provides clustering based on connectivity using the METIS package.
    Adapted from: 

    Hierarchical Transformer for Scalable Graph Learning
    https://arxiv.org/abs/2305.02866
    """
    def __init__(self, num_nodes: List[int]):
        self.num_layers = len(num_nodes)
        self.num_nodes = num_nodes

    def __call__(
        self, 
        data: torch_geometric.data.Data, 
        dataset_name: Optional[str] = None) -> List[Tensor]:
        """
        Creates the hierarchy maps used to coarsen the Data objects. If the Generator
        finds existing partition files, it will load them instead of recomputing the 
        partition to save time.

        Args:
            data: torch_geometric.data.Data, input graph
            dataset_name: Optional[str], name of dataset, used for saving partition
        Returns:
            hier_map: List[Tensor], list of cluster maps for each layer
        """

        save_path = None
        if dataset_name is not None:
            folder_path = f"./partitions/metis_{dataset_name[0]}/{dataset_name[1]}"
            save_path = f"{folder_path}/{dataset_name[2]}_{self.num_nodes}.pt"
            if os.path.exists(save_path):
                hier_map = torch.load(save_path)
                
                if VERBOSE:
                    print(f"Loaded saved partition file {save_path}.")
                return hier_map

        if self.num_layers == 0:
            return []
        
        if self.num_layers == [-1]:
            return [torch.zeros(data.num_nodes, dtype=int)]


        t1 = time.time() # init block
        edge_index = data.edge_index
        transform = T.to_sparse_tensor.ToSparseTensor()
        _copied_data = copy.copy(data)
        t1 = time.time()
        adj_t = transform(_copied_data).adj_t
        adj_t = adj_t.to_symmetric()
        part_fn = torch.ops.torch_sparse.partition
        counter = time.perf_counter()
        hier_map = []

        for i in range(self.num_layers):
            if self.num_nodes[i] == 1:
                if i == 0:
                    hier_map.append(torch.zeros(data.num_nodes, dtype=int))
                else:
                    hier_map.append(torch.zeros(hier_map[-1].max()+1, dtype=int))
                continue

            row_ptr, col, _ = adj_t.csr()
            cluster_map = part_fn(row_ptr, col, None, self.num_nodes[i], False)
            cluster_map = rearrange_cluster_map(cluster_map)
            hier_map.append(cluster_map)

            edge_index, _ = pool_edge(cluster=cluster_map, edge_index=edge_index.contiguous())

            # failsafe incase too few non-empty clusters are created
            if hier_map[i].max() + 1 < self.num_nodes[i]:
                self.num_nodes[i] = hier_map[-1].max() + 1

            next_data = transform(torch_geometric.data.Data(edge_index=edge_index, num_nodes=self.num_nodes[i]))
            adj_t = next_data.adj_t
            adj_t = adj_t.to_symmetric()

        if save_path is not None and self.num_layers > 0:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            torch.save(hier_map, save_path)


        return hier_map

