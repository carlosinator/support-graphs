from .data import HierarchicalGraphData, HierarchyFeatureCollator
from .coarsener import HierarchyGenerator

from graphgps.transform.posenc_stats import compute_posenc_stats
import numpy as np
import copy
import torch
import time
from torch_geometric.utils import index_to_mask, to_undirected
from torch_geometric.data import Data
from tqdm import tqdm
from torch_geometric.graphgym.loader import set_dataset_info

import logging
import os.path as osp
import time
from functools import partial

import numpy as np
import torch
import torch_geometric.transforms as T
from numpy.random import default_rng
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import (Actor, GNNBenchmarkDataset, Planetoid,
                                      TUDataset, WebKB, WikipediaNetwork, ZINC,
                                      Flickr)
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loader import load_pyg, load_ogb, set_dataset_attr
from torch_geometric.graphgym.register import register_loader
from torch_geometric.data.batch import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.loader.neighbor_loader import NeighborLoader

from graphgps.loader.dataset.aqsol_molecules import AQSOL
from graphgps.loader.dataset.coco_superpixels import COCOSuperpixels
from graphgps.loader.dataset.malnet_tiny import MalNetTiny
from graphgps.loader.dataset.voc_superpixels import VOCSuperpixels
from graphgps.loader.split_generator import (prepare_splits,
                                             set_dataset_splits)
from graphgps.transform.posenc_stats import compute_posenc_stats
from graphgps.transform.task_preprocessing import task_specific_preprocessing
from graphgps.transform.transforms import (pre_transform_in_memory,
                                           typecast_x, concat_x_and_pos,
                                           clip_graphs_to_size)

from graphgps.loader.master_loader import (preformat_GNNBenchmarkDataset,
                                        preformat_MalNetTiny,
                                        preformat_TUDataset,
                                        preformat_ZINC,
                                        preformat_AQSOL,
                                        preformat_VOCSuperpixels,
                                        preformat_COCOSuperpixels,
                                        preformat_OGB_Graph,
                                        preformat_OGB_PCQM4Mv2,
                                        preformat_Peptides,
                                        preformat_PCQM4Mv2Contact,
                                        log_loaded_dataset)

def get_dataset_from_cfg(format, name, dataset_dir):
    if format.startswith('PyG-'):
        pyg_dataset_id = format.split('-', 1)[1]
        dataset_dir = osp.join(dataset_dir, pyg_dataset_id)

        print(f"Loading PyG dataset {pyg_dataset_id} from {dataset_dir}")

        if pyg_dataset_id == 'Actor':
            if name != 'none':
                raise ValueError(f"Actor class provides only one dataset.")
            dataset = Actor(dataset_dir)

        elif pyg_dataset_id == 'GNNBenchmarkDataset':
            dataset = preformat_GNNBenchmarkDataset(dataset_dir, name)

        elif pyg_dataset_id == 'MalNetTiny':
            dataset = preformat_MalNetTiny(dataset_dir, feature_set=name)

        elif pyg_dataset_id == 'Planetoid':
            dataset = Planetoid(dataset_dir, name)

        elif pyg_dataset_id == 'TUDataset':
            dataset = preformat_TUDataset(dataset_dir, name)

        elif pyg_dataset_id == 'WebKB':
            dataset = WebKB(dataset_dir, name)

        elif pyg_dataset_id == 'WikipediaNetwork':
            if name == 'crocodile':
                raise NotImplementedError(f"crocodile not implemented")
            dataset = WikipediaNetwork(dataset_dir, name,
                                    geom_gcn_preprocess=True)

        elif pyg_dataset_id == 'ZINC':
            dataset = preformat_ZINC(dataset_dir, name)
            
        elif pyg_dataset_id == 'AQSOL':
            dataset = preformat_AQSOL(dataset_dir, name)

        elif pyg_dataset_id == 'VOCSuperpixels':
            dataset = preformat_VOCSuperpixels(dataset_dir, name,
                                            cfg.dataset.slic_compactness)

        elif pyg_dataset_id == 'COCOSuperpixels':
            dataset = preformat_COCOSuperpixels(dataset_dir, name,
                                                cfg.dataset.slic_compactness)
        
        elif pyg_dataset_id == 'Flickr':
            dataset = Flickr(dataset_dir)

        else:
            raise ValueError(f"Unexpected PyG Dataset identifier: {format}")

    # GraphGym default loader for Pytorch Geometric datasets
    elif format == 'PyG':
        dataset = load_pyg(name, dataset_dir)

    elif format == 'OGB':
        if name.startswith('ogbg'):
            dataset = preformat_OGB_Graph(dataset_dir, name.replace('_', '-'))

        elif name.startswith('PCQM4Mv2-'):
            subset = name.split('-', 1)[1]
            dataset = preformat_OGB_PCQM4Mv2(dataset_dir, subset)

        elif name.startswith('peptides-'):
            dataset = preformat_Peptides(dataset_dir, name)

        ### Link prediction datasets.
        elif name.startswith('ogbl-'):
            # GraphGym default loader.
            dataset = load_ogb(name, dataset_dir)
            # OGB link prediction datasets are binary classification tasks,
            # however the default loader creates float labels => convert to int.
            def convert_to_int(ds, prop):
                tmp = getattr(ds.data, prop).int()
                set_dataset_attr(ds, prop, tmp, len(tmp))
            convert_to_int(dataset, 'train_edge_label')
            convert_to_int(dataset, 'val_edge_label')
            convert_to_int(dataset, 'test_edge_label')

        elif name.startswith('PCQM4Mv2Contact-'):
            dataset = preformat_PCQM4Mv2Contact(dataset_dir, name)

        elif name == 'ogbn-arxiv':
            dataset = PygNodePropPredDataset("ogbn-arxiv", root='data/datasets/tests')
            splits = dataset.get_idx_split()
            split_names = ['train_mask', 'val_mask', 'test_mask']
            for i, key in enumerate(splits.keys()):
                mask = index_to_mask(splits[key], size=dataset._data.y.shape[0])
                set_dataset_attr(dataset, split_names[i], mask, len(mask))
            edge_index = to_undirected(dataset._data.edge_index)
            set_dataset_attr(dataset, 'edge_index', edge_index,
                            edge_index.shape[1])

        else:
            raise ValueError(f"Unsupported OGB(-derived) dataset: {name}")
    else:
        raise ValueError(f"Unknown data format: {format}")

    pre_transform_in_memory(dataset, partial(task_specific_preprocessing, cfg=cfg))

    log_loaded_dataset(dataset, format, name)

    return dataset


class DataModule:
    """
    Base class DataModule, which loads the dataset
    and performs the coarsening.
    """
    def __init__(self, cfg):
        """
        Initialize DataModule with the given configuration.
        Args:
            cfg: Configuration dict.
        """
        self.cfg = cfg
        # Precompute necessary statistics for positional encodings.
        self.pe_enabled_list = []
        for key, pecfg in cfg.items():
            if key.startswith('posenc_') and pecfg.enable:
                pe_name = key.split('_', 1)[1]
                self.pe_enabled_list.append(pe_name)
                if hasattr(pecfg, 'kernel'):
                    # Generate kernel times if functional snippet is set.
                    if pecfg.kernel.times_func:
                        pecfg.kernel.times = list(eval(pecfg.kernel.times_func))
                    logging.info(f"Parsed {pe_name} PE kernel times / steps: "
                                f"{pecfg.kernel.times}")
        

        # load in data
        self.load_dataset(cfg)

        if self.pe_enabled_list:
            print("loading posencs...")
            for i in tqdm(range(len(self.dataset))):
                self.dataset[i] = self.compute_posenc(self.dataset[i])

        return

    def load_dataset(self, cfg):
        """
        Load dataset and perform coarsening.
        Args:
            cfg: Configuration dict.
        """
        format = cfg.dataset.format
        name = cfg.dataset.name
        dataset_dir = cfg.dataset.dir
        coarsen_method = cfg.hsg.coarsen_method

        # quick load functionality
        save_path = f"./partitions/fullds_{coarsen_method}_{format}_{name}_{cfg.hsg.num_hierarchy_nodes}.pt"
        if osp.exists(save_path):
            print("Loading saved partition file from ", save_path)
            t1 = time.time()
            load_dict = torch.load(save_path)
            print(f"Done. took {time.time() - t1:.4f} seconds.")
            self.dataset = load_dict["dataset"]
            self.orig_split_idxs = load_dict.get("orig_split_idxs", None)
            self.orig_train_mask = load_dict.get("orig_train_mask", None)
            self.orig_val_mask = load_dict.get("orig_val_mask", None)
            self.orig_test_mask = load_dict.get("orig_test_mask", None)

            # set_dataset_info(self.dataset[0])
            cfg.share = load_dict.get("cfg_share", cfg.share)
            return

        # no such dir, continue function: 
        # load dataset
        dataset = get_dataset_from_cfg(format, name, dataset_dir)
        self.raw_data = dataset
        if cfg.dataset.transductive:
            self.raw_train_dataset = copy.deepcopy(self.raw_data)
        
        self.dataset = []

        set_dataset_info(dataset)
        cfg.share.num_splits = 3


        print(f"Starting {cfg.hsg.coarsen_method} partitioning...")
        t1 = time.time()
        for i in tqdm(range(len(self.raw_data))):
            
            num_hier_nodes = self._instantiate_coarsening(
                self.raw_data[i].num_nodes, 
                cfg.hsg.num_hierarchy_nodes)

            # num_hier_nodes = cfg.hsg.num_hierarchy_nodes
            self.dataset.append(HierarchicalGraphData(
                graph_data=self.raw_data[i],
                hierarchy_generator=HierarchyGenerator(
                    generator_type=coarsen_method,
                    num_nodes=num_hier_nodes,
                    dataset_name=(format, name, i),),
                feature_collator=HierarchyFeatureCollator(
                    collator_type='mean')
            ))
        print("Elapsed time: ", time.time() - t1)

        # save to dir
        load_dict = {}
        load_dict["dataset"] = self.dataset
        if cfg.dataset.transductive:
            self.orig_train_mask = self.raw_data.train_mask
            self.orig_val_mask = self.raw_data.val_mask
            self.orig_test_mask = self.raw_data.test_mask
            load_dict["orig_train_mask"] = self.orig_train_mask
            load_dict["orig_val_mask"] = self.orig_val_mask
            load_dict["orig_test_mask"] = self.orig_test_mask

        else:
            self.orig_split_idxs = self.raw_data.split_idxs
            load_dict["orig_split_idxs"] = self.orig_split_idxs
        
        load_dict["cfg_share"] = cfg.share
        torch.save(load_dict, save_path)

        # set_dataset_info(self.dataset[0])

        return

    def _instantiate_coarsening(self, num_real_nodes, coarsening):
        """ 
        helper function that converts the coarsening list 
        into a list of number of nodes in each hierarchy. 

        Args:
            num_real_nodes: number of nodes in the original graph
            coarsening: list of coarsening factors
        Returns:
            num_parents: list of number of nodes in each hierarchy
        """
        if len(coarsening) == 0:
            return []

        num_parents = []
        prev_num = num_real_nodes
        for coar in coarsening:
            if coar == -1:
                num_parents.append(1)
                break
            num_parents.append(max(1, int(prev_num * coar)))
            prev_num = num_parents[-1]

        return num_parents
        
    def compute_posenc(self, dataset):
        """ wrapper for compute_posenc_stats (normal graphgps function) """
        cfg = self.cfg
        if not self.pe_enabled_list:
            print("posencs not enabled.")
            return dataset

        for i in range(dataset.num_hierarchies):
            dataset.hier_data[i] = compute_posenc_stats(
                dataset.hier_data[i], self.pe_enabled_list, True, cfg
            )

        return dataset


class SimpleDataModule(DataModule):
    """
    Extends DataModule and implements coarsening post-processing compatible
    with the tested datasets:
    (PascalVOC-SP, COCO-SP, Peptides-func, Peptides-struct, ogbg-molpcba)
    
    Requires some adaptation for training on transductive tasks, some code is
    available from:
    Hierarchical Transformer for Scalable Graph Learning
    https://arxiv.org/abs/2305.02866 
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # detect attributes in dataset
        self.detect_attributes()

        self.add_one_hot_attr()

        self.compile_into_data()
        return

    def add_one_hot_attr(self):
        """ adds node & edge one hot attributes to data objects """
        hierarchy_depth = len(self.dataset[0].hier_data)
        for i in range(len(self.dataset)):
            for j in range(hierarchy_depth):
                one_hot_hier = j * torch.ones(
                    self.dataset[i].hier_data[j].x.size(0), dtype=torch.long)
                self.dataset[i].hier_data[j].node_onehot = one_hot_hier

                edge_onehot = j * torch.ones(
                    self.dataset[i].hier_data[j].edge_index.size(1), 
                    dtype=torch.long)
                self.dataset[i].hier_data[j].edge_onehot = edge_onehot
        
        return

    def detect_attributes(self):
        """ 
        detects attributes relevant to the studies graphs 
        and adds them back to the final data object.
        """
        example_data =  self.dataset[0].hier_data[0]
        self.attr_list = []
        possible_attrs = [
            "x", "edge_index", "edge_attr", "y", 
            "train_mask", "val_mask", "test_mask", "edge_index_labeled"]
        for attr in possible_attrs:
            if attr in example_data:
                self.attr_list.append(attr)

        # add virtual attributes
        self.attr_list.append("fit_mask")
        self.attr_list.append("real_mask")
        self.attr_list.append("real_edge_mask")

        # add one hot attributes
        self.attr_list.append("node_onehot")
        self.attr_list.append("edge_onehot")

        # add posenc attributes
        if "LapPE" in self.pe_enabled_list:
            self.attr_list.append("EigVals")
            self.attr_list.append("EigVecs")
        if "RWSE" in self.pe_enabled_list:
            self.attr_list.append("pestat_RWSE")

        return

    def compile_hier_edge_index(self, hier_map):
        """
        compiles vertical edges into one edge_index object

        Args:
            hier_map: list of mapping tensors for each hierarchy
        Returns:
            full_hier_edge_index: compiled edge_index
            full_hier_edge_attr: compiled edge_attr
            full_mask: compiled mask
        """
        if "edge_attr" in self.dataset[0].hier_data[0]:
            ea_size = self.dataset[0].hier_data[0].edge_attr.size(1)
        else:
            ea_size = 0

        ei_lst = []
        ea_lst = []
        ei_msks = []
        offset = 0
        for i in range(len(hier_map)):
            # careful creation of vertical edges, requires correct offsets
            diff = hier_map[i].size(0)
            this_ei = torch.cat((
                offset + torch.arange(diff)[None, :], 
                hier_map[i][None, :] + diff + offset), dim=0)
            offset += diff
            ei_lst.append(this_ei)
            ea_lst.append(torch.zeros((this_ei.size(1), ea_size)))
            ei_msks.append(i * torch.ones(this_ei.size(1), dtype=torch.long))

        # if no hierarchical edges, create empty tensors
        if len(ei_lst) == 0:
            full_hier_edge_index = torch.zeros((2,0), dtype=torch.long)
            full_hier_edge_attr = torch.zeros((0, ea_size))
            full_mask = torch.zeros((0,), dtype=torch.bool)
        else:
            full_hier_edge_index = torch.cat(ei_lst, dim=1)
            full_hier_edge_attr = torch.cat(ea_lst, dim=0)
            full_mask = torch.cat(ei_msks, dim=0)

        # make undirected
        undir_hier_edge_index, full_mask =\
            to_undirected(full_hier_edge_index, full_mask)
        undir_hier_edge_index, full_hier_edge_attr =\
            to_undirected(full_hier_edge_index, full_hier_edge_attr)        
        return undir_hier_edge_index, full_hier_edge_attr, full_mask

    def connect_graph_hiers(self, isolated_graph, hier_map):
        """
        calls compile_hier_edge_index, adds hierarchy based edge masks, 
        and connects the graph

        Args:
            isolated_graph: isolated graph object
            hier_map: list of mapping tensors for each hierarchy
        Returns:
            isolated_graph: connected graph object
        """

        # compile hier_map into valid attributes
        hier_ei, hier_ea, hier_mask = self.compile_hier_edge_index(hier_map)

        # add hierarchical edges
        isolated_graph.edge_index = torch.cat((
            isolated_graph.edge_index, 
            hier_ei), dim=1)
        if "edge_attr" in self.attr_list:
            dt = isolated_graph.edge_attr.dtype
            isolated_graph.edge_attr = torch.cat((
                isolated_graph.edge_attr, 
                hier_ea), dim=0)
            isolated_graph.edge_attr = isolated_graph.edge_attr.to(dt)

        if "real_edge_mask" in self.attr_list:
            isolated_graph.real_edge_mask = torch.cat((
                isolated_graph.real_edge_mask, 
                torch.zeros(hier_ei.size(1), dtype=torch.bool)), dim=0)

        if "edge_index_labeled" in self.attr_list:
            isolated_graph.edge_index_labeled = torch.cat((
                isolated_graph.edge_index_labeled, 
                torch.zeros(hier_ei.size(1), dtype=torch.bool)), dim=0)
        
        if "edge_onehot" in self.attr_list:
            fill_in = len(hier_map) + 1
            isolated_graph.edge_onehot = torch.cat((
                isolated_graph.edge_onehot, 
                fill_in + hier_mask), dim=0)

        return isolated_graph

    def create_data_object(self, graph_data):
        """
        creates a training-ready data object by adding all necessary
        attributes and calling connect_graph_hiers.

        Args:
            graph_data: HierarchicalGraphData object
        Returns:
            graph: Data object (PyG Data object ready for training)
        """

        assert hasattr(self, "attr_list"), \
            "Attributes not detected yet, plase call detect_attributes first"

        datalist = graph_data.hier_data
        real_data = datalist[0]
        if "fit_mask" in self.attr_list:
            real_data.fit_mask = torch.ones(
                real_data.num_nodes, dtype=torch.bool)
        if "real_mask" in self.attr_list:
            real_data.real_mask = torch.ones(
                real_data.num_nodes, dtype=torch.bool)
        if "real_edge_mask" in self.attr_list:
            real_data.real_edge_mask = torch.ones(
                real_data.edge_index.size(1), dtype=torch.bool)
        
        sublist = [real_data]
        for i in range(1, len(datalist)):
            data = datalist[i]

            if "y" in self.attr_list and "y" not in data:
                data.y = - torch.ones(data.num_nodes, dtype=torch.long)

            if real_data.y.dim() > 1: # TODO: quick fix for multi-tasking graph tasks
                data.y = torch.zeros((0, real_data.y.size(1)))

            if "train_mask" in self.attr_list:
                data.train_mask = torch.zeros(
                    data.num_nodes, dtype=torch.bool)
                data.val_mask = torch.zeros(
                    data.num_nodes, dtype=torch.bool)
                data.test_mask = torch.zeros(
                    data.num_nodes, dtype=torch.bool)

            if "edge_attr" in self.attr_list and "edge_attr" not in data:
                data.edge_attr = torch.zeros(
                    (data.edge_index.size(1), real_data.edge_attr.size(1)))
            if "real_mask" in self.attr_list and "real_mask" not in data:
                data.real_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            if "fit_mask" in self.attr_list and "fit_mask" not in data:
                data.fit_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            if "real_edge_mask" in self.attr_list and "real_edge_mask" not in data:
                data.real_edge_mask = torch.zeros(
                    data.edge_index.size(1), dtype=torch.bool)

            sublist.append(data)

        # generate a graph from the sublist
        gen_batch = Batch.from_data_list(sublist, exclude_keys=None)
        graph = Data(x=gen_batch.x, edge_index=gen_batch.edge_index)
        
        for attr in self.attr_list: # add all detected attributes
            setattr(graph, attr, getattr(gen_batch, attr))

        # connect hierarchical edges
        graph = self.connect_graph_hiers(graph, graph_data.hier_map)

        return graph

    def compile_into_data(self):
        """
        wraps create_data_object into a loop and compiles the dataset
        """
        new_data_list = []
        print("compiling into data...")
        for data in tqdm(self.dataset):            
            new_data_list.append(self.create_data_object(data))
            
        self.simple_dataset = new_data_list

        return
    
    def get_loaders(self, cfg):
        """
        returns dataloaders for train, val, and test based on the task.
        Requires adaptation for transductive tasks.
        """

        loaders = []
        masks = ['train', 'val', 'test']
        if cfg.train.batch_size > 0:
            batch_size = cfg.train.batch_size
        else:
            batch_size = len(self.simple_dataset)

        for i in range(cfg.share.num_splits):
            split_dataset = [self.simple_dataset[j] for j in self.orig_split_idxs[i]]
            loader = DataLoader(
                split_dataset, 
                batch_size=cfg.train.batch_size,
                shuffle=True if i == 0 else False,
            )
            loaders.append(loader)

        self.train_loader = loaders[0]
        self.val_loader = loaders[1]
        self.test_loader = loaders[2]

        return loaders