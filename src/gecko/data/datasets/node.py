from __future__ import annotations

# COMMON
from typing import Callable
from typing import Optional
import datetime
import os
import torch
import numpy as np
import networkx as nx
import torch.nn.functional as F
import tqdm
import pickle
import json
from itertools import chain

# FOR DGL DATASETS
import dgl
from dgl.data.utils import save_graphs
from dgl.data.utils import load_graphs
from dgl.data.utils import save_info
from dgl.data.utils import load_info
from dgl.data.utils import makedirs
from dgl.data.utils import _get_dgl_url
from dgl.data.utils import download
from dgl.data.utils import extract_archive
from ogb.graphproppred import DglGraphPropPredDataset

# for aromaticity dataset
from dgllife.data import PubChemBioAssayAromaticity
from dgllife.data.csv_dataset import MoleculeCSVDataset
from dgllife.utils.mol_to_graph import smiles_to_bigraph
import pandas as pd



import torch
import dgl
import os
import pickle
from dgl.data.utils import download
from ogb.nodeproppred import DglNodePropPredDataset
from torch_scatter import scatter


def load_node_dataset(dataset_name, dataset_load_func, incr_type, save_path):
    """
        The function for load node-level datasets.
    """
    cover_rule = {'feat': 'node', 'label': 'node', 'train_mask': 'node', 'val_mask': 'node', 'test_mask': 'node'}
    if dataset_load_func is not None:
        custom_dataset = dataset_load_func(save_path=save_path)
        graph = custom_dataset['graph']
        num_feats = custom_dataset['num_feats']
        num_classes = custom_dataset['num_classes']
    elif dataset_name in ['cora'] and incr_type in ['task', 'class']:
        dataset = dgl.data.CoraGraphDataset(raw_dir=save_path, verbose=False)
        graph = dataset._g
        num_feats, num_classes = graph.ndata['feat'].shape[-1], dataset.num_classes
    elif dataset_name in ['citeseer'] and incr_type in ['task', 'class']:
        dataset = dgl.data.CiteseerGraphDataset(raw_dir=save_path)
        graph = dataset._g
        num_feats, num_classes = graph.ndata['feat'].shape[-1], dataset.num_classes
    elif dataset_name in ['corafull'] and incr_type in ['task', 'class']:
        dataset = dgl.data.CoraFullDataset(raw_dir=save_path, verbose=False)
        graph = dataset._graph
        num_feats, num_classes = graph.ndata['feat'].shape[-1], dataset.num_classes
        
        # We need to designate train/val/test split since DGL does not provide the information.
        # We used random train/val/test split (6 : 2 : 2)
        pkl_path = os.path.join(save_path, f'corafull_metadata_allIL.pkl')
        if not os.path.isfile(pkl_path):
            download(f'https://github.com/ShinhwanKang/BeGin/raw/main/metadata/corafull_metadata_allIL.pkl', pkl_path)
        with open(pkl_path, 'rb') as handle:
            metadata = pickle.load(handle)
        inner_tvt_splits = metadata['inner_tvt_splits'] % 10
        graph.ndata['train_mask'] = (inner_tvt_splits < 6)
        graph.ndata['val_mask'] = (6 <= inner_tvt_splits) & (inner_tvt_splits < 8)
        graph.ndata['test_mask'] = (8 <= inner_tvt_splits)
        
    elif dataset_name in ['ogbn-arxiv'] and incr_type in ['task', 'class', 'time']:
        dataset = DglNodePropPredDataset('ogbn-arxiv', root=save_path)
        graph, label = dataset[0]
        num_feats, num_classes = graph.ndata['feat'].shape[-1], dataset.num_classes
        
        # to_bidirected
        srcs, dsts = graph.all_edges()
        graph.add_edges(dsts, srcs)
        
        if incr_type == 'time':
            # load train/val/test split
            pkl_path = os.path.join(save_path, f'ogbn-arxiv_metadata_timeIL.pkl')
            download(f'https://github.com/ShinhwanKang/BeGin/raw/main/metadata/ogbn-arxiv_metadata_timeIL.pkl', pkl_path)
            metadata = pickle.load(open(pkl_path, 'rb'))
            inner_tvt_splits = metadata['inner_tvt_splits']
            graph.ndata['train_mask'] = (inner_tvt_splits < 4)
            graph.ndata['val_mask'] = (inner_tvt_splits == 4)
            graph.ndata['test_mask'] = (inner_tvt_splits > 4)
            # time information for splitting tasks
            graph.ndata['time'] = torch.clamp(graph.ndata.pop('year').squeeze() - 1997, 0, 20000)
        else:
            # load train/val/test split
            split_idx = dataset.get_idx_split()
            for _split, _split_name in [('train', 'train'), ('valid', 'val'), ('test', 'test')]:
                _indices = torch.zeros(graph.num_nodes(), dtype=torch.bool)
                _indices[split_idx[_split]] = True
                graph.ndata[_split_name + '_mask'] = _indices
        
        # load target label and timestamp information
        graph.ndata['label'] = label.squeeze()
        
    elif dataset_name in ['ogbn-products'] and incr_type in ['task', 'class']:
        dataset = DglNodePropPredDataset('ogbn-products', root=save_path)
        graph, label = dataset[0]
        num_feats, num_classes = graph.ndata['feat'].shape[-1], dataset.num_classes
        
        # load train/val/test split
        split_idx = dataset.get_idx_split()
        for _split, _split_name in [('train', 'train'), ('valid', 'val'), ('test', 'test')]:
            _indices = torch.zeros(graph.num_nodes(), dtype=torch.bool)
            _indices[split_idx[_split]] = True
            graph.ndata[_split_name + '_mask'] = _indices
        
        # load target label and timestamp information
        graph.ndata['label'] = label.squeeze()
    elif dataset_name in ['ogbn-proteins'] and incr_type in ['domain']:
        dataset = DglNodePropPredDataset('ogbn-proteins', root=save_path)
        graph, label = dataset[0]
        
        # create node features using edge features + load species information
        # (See https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/proteins/gnn.py : commit d04eada)
        uefa_raw_edge_features = graph.edata.pop('feat')
        graph.ndata['feat'] = scatter(uefa_raw_edge_features, graph.edges()[0], dim=0, reduce='mean')
        graph.uefa_raw_edge_features = uefa_raw_edge_features
        unique_ids = torch.unique(graph.ndata['species'])
        raw_species_to_domain = -torch.ones(unique_ids.max().item() + 1, dtype=torch.long)
        raw_species_to_domain[unique_ids] = torch.arange(8)
        graph.ndata['species'] = raw_species_to_domain[graph.ndata.pop('species').squeeze(-1)]
        num_feats, num_classes = graph.ndata['feat'].shape[-1], label.shape[-1]
        
        # load train/val/test split
        pkl_path = os.path.join(save_path, f'ogbn-proteins_metadata_domainIL.pkl')
        if not os.path.exists(pkl_path):
            download(f'https://github.com/ShinhwanKang/BeGin/raw/main/metadata/ogbn-proteins_metadata_domainIL.pkl', pkl_path)
        with open(pkl_path, 'rb') as metadata_file:
            metadata = pickle.load(metadata_file)
        inner_tvt_splits = metadata['inner_tvt_splits']
        graph.ndata['train_mask'] = (inner_tvt_splits < 4)
        graph.ndata['val_mask'] = (inner_tvt_splits == 4)
        graph.ndata['test_mask'] = (inner_tvt_splits > 4)
        
        # load target label and domain information
        graph.ndata['label'] = label
        if incr_type == 'domain': graph.ndata['domain'] = graph.ndata.pop('species').squeeze()
        
    elif dataset_name in ['ogbn-mag'] and incr_type in ['task', 'class', 'time']:
        dataset = DglNodePropPredDataset('ogbn-mag', root=save_path)
        _graph, _label = dataset[0]
        srcs, dsts = _graph.edges(etype='cites')
        graph = dgl.graph((srcs, dsts))
        
        # pick nodes whose entity is 'paper'
        graph.ndata['feat'] = _graph.ndata['feat']['paper']
        graph.add_edges(dsts, srcs)
        label = _label['paper'].squeeze()
        
        split_idx = dataset.get_idx_split()
        
        # (for task, class) select classes with at least 10 nodes (in train, valid, and test split)
        if incr_type in ['task', 'class']:
            traincnt = torch.bincount(label[split_idx['train']['paper']])
            valcnt = torch.bincount(label[split_idx['valid']['paper']])
            testcnt = torch.bincount(label[split_idx['test']['paper']])
            considered_labels = torch.nonzero(torch.min(torch.stack((traincnt, valcnt, testcnt), dim=-1), dim=-1).values >= 10, as_tuple=True)[0]
            processed_labels = torch.ones(label.max() + 1, dtype=torch.long) * considered_labels.shape[0]
            processed_labels[considered_labels] = torch.arange(considered_labels.shape[0])
            label = processed_labels[label]
            num_feats, num_classes = graph.ndata['feat'].shape[-1], label.max().item() # ignore the last class
            
            # load train/val/test split
            for _split, _split_name in [('train', 'train'), ('valid', 'val'), ('test', 'test')]:
                _indices = torch.zeros(graph.num_nodes(), dtype=torch.bool)
                _indices[split_idx[_split]['paper']] = True
                graph.ndata[_split_name + '_mask'] = _indices
            graph.ndata['label'] = label.squeeze()
        elif incr_type in ['time']:
            pkl_path = os.path.join(save_path, f'ogbn-mag_metadata_timeIL.pkl')
            download(f'https://github.com/ShinhwanKang/BeGin/raw/main/metadata/ogbn-mag_metadata_timeIL.pkl', pkl_path)
            metadata = pickle.load(open(pkl_path, 'rb'))
            inner_tvt_splits = metadata['inner_tvt_splits']
            graph.ndata['train_mask'] = (inner_tvt_splits < 4)
            graph.ndata['val_mask'] = (inner_tvt_splits == 4)
            graph.ndata['test_mask'] = (inner_tvt_splits > 4)
            
            graph.ndata['label'] = label.squeeze()
            num_feats, num_classes = graph.ndata['feat'].shape[-1], (label.max().item() + 1)
            graph.ndata['time'] = _graph.ndata['year']['paper'].squeeze() - 2010
            
    elif dataset_name in ['twitch'] and incr_type in ['domain']:
        dataset = TwitchGamerNodeDataset('twitch', raw_dir=save_path)
        graph = dataset[0]
        num_feats, num_classes = graph.ndata['feat'].shape[-1], dataset.num_classes
        
        pkl_path = os.path.join(save_path, f'twitch_metadata_domainIL.pkl')
        download(f'https://github.com/ShinhwanKang/BeGin/raw/main/metadata/twitch_metadata_domainIL.pkl', pkl_path)
        metadata = pickle.load(open(pkl_path, 'rb'))
        inner_tvt_splits = metadata['inner_tvt_splits']
        graph.ndata['train_mask'] = (inner_tvt_splits < 4)
        graph.ndata['val_mask'] = (inner_tvt_splits == 4)
        graph.ndata['test_mask'] = (inner_tvt_splits > 4)
            
    else:
        raise NotImplementedError("Tried to load unsupported scenario.")
        
    # We hide information of unseen nodes (for Time-IL) 
    for k in graph.ndata.keys():
        if k not in cover_rule:
            cover_rule[k] = 'node'
    for k in graph.edata.keys():
        if k not in cover_rule:
            cover_rule[k] = 'edge'
    
    # remove and add self-loop (to prevent duplicates)
    srcs, dsts = graph.edges()
    is_non_loop = (srcs != dsts)
    final_graph = dgl.graph((srcs[is_non_loop], dsts[is_non_loop]), num_nodes=graph.num_nodes())
    for k in graph.ndata.keys():
        final_graph.ndata[k] = graph.ndata[k]
    for k in graph.edata.keys():
        final_graph.edata[k] = graph.edata[k][is_non_loop]
    final_graph = dgl.add_self_loop(final_graph)
    if hasattr(graph, 'uefa_raw_edge_features'):
        non_loop_features = graph.uefa_raw_edge_features[is_non_loop]
        loop_features = torch.zeros(
            graph.num_nodes(),
            non_loop_features.shape[-1],
            dtype=non_loop_features.dtype,
        )
        final_graph.uefa_context_edge_features = torch.cat(
            (non_loop_features, loop_features), dim=0
        )
        final_graph.uefa_context_edge_feature_valid_mask = torch.cat(
            (
                torch.ones(non_loop_features.shape[0], dtype=torch.bool),
                torch.zeros(graph.num_nodes(), dtype=torch.bool),
            ),
            dim=0,
        )
        # The aligned final tensor supersedes the raw dataset tensor. Keeping
        # both alive costs multiple GiB on OGBN-Proteins and serves no later
        # consumer.
        del non_loop_features, loop_features
        delattr(graph, 'uefa_raw_edge_features')
        del uefa_raw_edge_features
    
    print("=====CHECK=====")
    print("num_classes:", num_classes, ", num_feats:", num_feats)
    print("graph.ndata['train_mask']:", 'train_mask' in graph.ndata)
    print("graph.ndata['val_mask']:", 'val_mask' in graph.ndata)
    print("graph.ndata['test_mask']:", 'test_mask' in graph.ndata)
    print("graph.ndata['label']:", 'label' in graph.ndata)
    if incr_type == 'time':
        print("graph.ndata['time']:", 'time' in graph.ndata)
    if incr_type == 'domain':
        print("graph.ndata['domain']:", 'domain' in graph.ndata)
    print("===============")
    
    return num_classes, num_feats, final_graph, cover_rule




class TwitchGamerNodeDataset(dgl.data.DGLBuiltinDataset):
    _url = 'http://snap.stanford.edu/data/twitch_gamers.zip'

    def __init__(self, dataset_name, raw_dir=None, force_reload=False, verbose=False, transform=None):
        super(TwitchGamerNodeDataset, self).__init__(name='twitch',
                                                 url=self._url,
                                                 raw_dir=raw_dir,
                                                 force_reload=force_reload,
                                                 verbose=verbose)
    def process(self):
        self._graphs = []
        edgefile = os.path.join(self.save_path, 'large_twitch_edges.csv')
        edgedata = pd.read_csv(edgefile)
        nodefile = os.path.join(self.save_path, 'large_twitch_features.csv')
        nodedata = pd.read_csv(nodefile)
        lang_to_domain = {'CS': 0, 'DA': 1, 'DE': 2, 'EN': 3, 'ES': 4, 'FI': 5, 'FR': 6, 'HU': 7, 'IT': 8, 'JA': 9,
                          'KO': 10, 'NL': 11, 'NO': 12, 'PL': 13, 'PT': 14, 'RU': 15, 'SV': 16, 'TH': 17, 'TR': 18, 'ZH': 19,
                          'OTHER': 20}
        graph = dgl.graph((edgedata['numeric_id_1'].values, edgedata['numeric_id_2'].values))

        langs = nodedata['language'].values.tolist()
        graph.ndata['domain'] = torch.LongTensor([lang_to_domain[_lang] for _lang in langs])
        normalized_views = torch.FloatTensor(nodedata['views'].values / float(nodedata['views'].values.max()))
        matures = torch.FloatTensor(nodedata['mature'].values)
        lifetimes = torch.FloatTensor(nodedata['life_time'].values)
        is_deads = torch.FloatTensor(nodedata['dead_account'].values)
        graph.ndata['feat'] = torch.stack((normalized_views, matures, lifetimes, is_deads), dim=-1)
        graph.ndata['label'] = torch.LongTensor(nodedata['affiliate'].values)
        self._graphs.append(graph)

    def has_cache(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        return os.path.exists(graph_path)

    def save(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        save_graphs(graph_path, self.graphs)

    def load(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        self._graphs = load_graphs(graph_path)[0]

    @property
    def graphs(self):
        return self._graphs

    @property
    def num_classes(self):
        return 2

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, item):
        return self.graphs[item]

