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
from ogb.linkproppred import DglLinkPropPredDataset


def load_linkp_dataset(dataset_name, dataset_load_func, incr_type, save_path):
    neg_edges = {}
    is_bipartite = False
    if dataset_load_func is not None:
        custom_dataset = dataset_load_func(save_path=save_path)
        graph = custom_dataset['graph']
        num_feats = custom_dataset['num_feats']
        tvt_splits = custom_dataset['tvt_splits'].clone()
        neg_edges = custom_dataset['neg_edges']
        tvt_splits[tvt_splits == 1] = 8
        tvt_splits[tvt_splits == 2] = 9
    elif dataset_name in ['ogbl-collab'] and incr_type in ['time']:
        dataset = DglLinkPropPredDataset('ogbl-collab', root=save_path)
        # load edges and negative edges
        split_edge = dataset.get_edge_split()
        train_graph = dataset[0]
        combined = {}
        for k in split_edge["train"].keys():
            combined[k] = torch.cat((split_edge["train"][k], split_edge["valid"][k], split_edge["test"][k]), dim=0)
            original = combined[k]
            if k == 'edge':
                rev_edges = torch.cat((combined['edge'][:, 1:2], combined['edge'][:, 0:1]), dim=-1)
                combined[k] = torch.cat((combined[k], rev_edges), dim=-1).view(-1, 2)
            else:
                combined[k] = torch.repeat_interleave(combined[k], 2, dim=0)
        
        # generate graphs with all edges (including val/test)
        graph = dgl.graph((combined['edge'][:, 0], combined['edge'][:, 1]), num_nodes=train_graph.num_nodes())
        for k in combined.keys():
            if k != 'edge':
                if k == 'year': graph.edata['time'] = torch.clamp(combined[k] - 1970, 0, 20000)
                else: graph.edata[k] = combined[k]
        for k in train_graph.ndata.keys():
            graph.ndata[k] = train_graph.ndata[k]
        _srcs, _dsts = map(lambda x: x.numpy().tolist(), graph.edges())
        edgeset = {(s, d) for s, d in zip(_srcs, _dsts)}
        
        num_feats = graph.ndata['feat'].shape[-1]
        # load time split and train/val/test split information
        pkl_path = os.path.join(save_path, f'ogbl-collab_metadata_timeIL.pkl')
        download(f'https://github.com/ShinhwanKang/BeGin/raw/main/metadata/ogbl-collab_metadata_timeIL.pkl', pkl_path)
        metadata = pickle.load(open(pkl_path, 'rb'))
        tvt_splits = metadata['inner_tvt_splits']    
        # choose negative edges
        neg_edges['val'] = torch.LongTensor([[_s, _d] for _s, _d in zip(*zip(*split_edge['valid']['edge_neg'].numpy().tolist())) if (_s, _d) not in edgeset])
        neg_edges['test'] = torch.LongTensor([[_s, _d] for _s, _d in zip(*zip(*split_edge['test']['edge_neg'].numpy().tolist())) if (_s, _d) not in edgeset])
        
    elif dataset_name in ['wikics'] and incr_type in ['domain']:
        dataset = WikiCSLinkDataset(raw_dir=save_path)
        graph = dataset._g
        num_feats = graph.ndata['feat'].shape[-1]
        # load tvt_splits and negative edges
        pkl_path = os.path.join(save_path, f'wikics_metadata_domainIL.pkl')
        download(f'https://github.com/ShinhwanKang/BeGin/raw/main/metadata/wikics_metadata_domainIL.pkl', pkl_path)
        metadata = pickle.load(open(pkl_path, 'rb'))
        tvt_splits = metadata['inner_tvt_splits']
        neg_edges = metadata['neg_edges']
        
        num_tasks = 54
        task_map = torch.LongTensor([[0,1,2,3,4,5,-1,6,7,8],
                                     [-1,9,10,11,12,13,14,15,16,17],
                                     [-1,-1,18,19,20,21,22,23,24,25],
                                     [-1,-1,-1,26,27,28,29,30,31,32],
                                     [-1,-1,-1,-1,33,34,35,36,37,38],
                                     [-1,-1,-1,-1,-1,39,40,41,42,43],
                                     [-1,-1,-1,-1,-1,-1,44,45,46,47],
                                     [-1,-1,-1,-1,-1,-1,-1,48,49,50],
                                     [-1,-1,-1,-1,-1,-1,-1,-1,51,52],
                                     [-1,-1,-1,-1,-1,-1,-1,-1,-1,53]])
        domain_info = graph.ndata.pop('domain')
        srcs, dsts = graph.edges()
        graph.edata['domain'] = task_map[torch.min(domain_info[srcs], domain_info[dsts]), torch.max(domain_info[srcs], domain_info[dsts])]
    elif dataset_name in ['askubuntu'] and incr_type in ['time']:
        dataset = AskUbuntuDataset(dataset_name=dataset_name, raw_dir=save_path)
        graph = dataset.graphs[0]
        num_feats = graph.ndata['feat'].shape[-1]
        
        pkl_path = os.path.join(save_path, f'askubuntu_metadata_timeIL.pkl')
        download(f'https://github.com/ShinhwanKang/BeGin/raw/main/metadata/askubuntu_metadata_timeIL.pkl', pkl_path, overwrite=False)
        metadata = pickle.load(open(pkl_path, 'rb'))
        tvt_splits = torch.repeat_interleave(metadata['inner_tvt_splits'], 2, dim=0)
        neg_edges = metadata['neg_edges']
    elif dataset_name in ['facebook'] and incr_type in ['domain']:
        dataset = FacebookLinkDataset(dataset_name=dataset_name, raw_dir=save_path)
        graph = dataset.graphs[0]
        num_feats = graph.ndata['feat'].shape[-1]
        pkl_path = os.path.join(save_path, f'facebook_metadata_domainIL.pkl')
        download(f'https://github.com/ShinhwanKang/BeGin/raw/main/metadata/facebook_metadata_domainIL.pkl', pkl_path, overwrite=False)
        metadata = pickle.load(open(pkl_path, 'rb'))
        tvt_splits = torch.repeat_interleave(metadata['inner_tvt_splits'], 2, dim=0)
        neg_edges = metadata['neg_edges']
    elif dataset_name in ['gowalla'] and incr_type in ['time']:
        is_bipartite = True
        num_srcs, num_dsts = 29858, 40981
        num_feats = 2
        pkl_path = os.path.join(save_path, f'gowalla_metadata_timeIL.pkl')
        download(f'https://github.com/ShinhwanKang/BeGin/raw/main/metadata/gowalla_metadata_timeIL.pkl', pkl_path, overwrite=False)
        metadata = pickle.load(open(pkl_path, 'rb'))
        srcs, dsts, _ = zip(*metadata['edges'])
        srcs, dsts = torch.LongTensor(srcs), (torch.LongTensor(dsts) + num_srcs)
        graph = dgl.graph((torch.stack((srcs, dsts), dim=-1).view(-1), torch.stack((dsts, srcs), dim=-1).view(-1)))
        feats = torch.zeros(num_srcs + num_dsts, 2)
        feats[:num_srcs, 0] = 1.
        feats[num_srcs:, 1] = 1.
        graph.ndata['feat'] = feats
        tvt_splits = torch.repeat_interleave(metadata['inner_tvt_splits'], 2, dim=0)
        graph.edata['time'] = torch.repeat_interleave(metadata['time'], 2, dim=0)
        neg_edges = metadata['neg_edges']
        neg_edges['val'][:, 1] += num_srcs
        neg_edges['test'][:, 1] += num_srcs
    elif dataset_name in ['movielens'] and incr_type in ['time']:
        is_bipartite = True
        dataset = MovielensDataset(dataset_name=dataset_name, raw_dir=save_path)
        num_srcs, num_dsts = 6040, 3952
        pkl_path = os.path.join(save_path, f'movielens_metadata_timeIL.pkl')
        download(f'https://github.com/ShinhwanKang/BeGin/raw/main/metadata/movielens_metadata_timeIL.pkl', pkl_path, overwrite=False)
        metadata = pickle.load(open(pkl_path, 'rb'))
        # edges, feats, time are not in preprocessed file (due to the license)
        metadata['edges'] = dataset.metadata['edges']
        metadata['feats'] = dataset.metadata['feats']
        metadata['time'] = dataset.metadata['time']
        
        num_feats = metadata['feats'][0].shape[-1] + metadata['feats'][1].shape[-1]
        srcs, dsts, _ = zip(*metadata['edges'])
        metadata['feats'] = list(map(torch.FloatTensor, metadata['feats']))
        srcs, dsts = torch.LongTensor(srcs) - 1, (torch.LongTensor(dsts) - 1 + num_srcs)
        graph = dgl.graph((torch.stack((srcs, dsts), dim=-1).view(-1), torch.stack((dsts, srcs), dim=-1).view(-1)))
        ufeats = torch.cat((metadata['feats'][0], torch.zeros(metadata['feats'][0].shape[0], metadata['feats'][1].shape[-1])), dim=-1)
        ifeats = torch.cat((torch.zeros(metadata['feats'][1].shape[0], metadata['feats'][0].shape[-1]), metadata['feats'][1]), dim=-1)
        graph.ndata['feat'] = torch.cat((ufeats, ifeats), dim=0)
        tvt_splits = torch.repeat_interleave(metadata['inner_tvt_splits'], 2, dim=0)
        graph.edata['time'] = torch.repeat_interleave(metadata['time'], 2, dim=0)

        neg_srcs, neg_dsts, _ = zip(*metadata['neg_edges'])
        negs = torch.stack((torch.LongTensor(neg_srcs), torch.LongTensor(neg_dsts) + num_srcs), dim=-1) - 1
        neg_edges = {'val': negs[0::2], 'test': negs[1::2]}
    else:
        raise NotImplementedError("Tried to load unsupported scenario.")
    
    print("=====CHECK=====")
    print("num_feats:", num_feats)
    print("inner_tvt_splits:", tvt_splits.shape)
    print("neg_edges['val']:", neg_edges['val'].shape)
    print("neg_edges['test']:", neg_edges['test'].shape)
    if incr_type == 'time':
        print("graph.edata['time']", graph.edata['time'].shape)
    if incr_type == 'domain':
        print("graph.edata['domain']", graph.edata['domain'].shape)
    print("===============")
    
    return num_feats, graph, tvt_splits, neg_edges, is_bipartite


def load_linkc_dataset(dataset_name, dataset_load_func, incr_type, save_path):
    if dataset_load_func is not None:
        custom_dataset = dataset_load_func(save_path=save_path)
        graph = custom_dataset['graph']
        num_feats = custom_dataset['num_feats']
        num_classes = custom_dataset['num_classes']
    elif dataset_name == 'bitcoin' and incr_type in ['task', 'class', 'domain', 'time']:
        dataset = BitcoinOTCDataset(dataset_name, raw_dir=save_path)
        graph = dataset[0]
        num_feats = graph.ndata['feat'].shape[-1]
        if incr_type == 'time':
            num_classes = 1
            num_tasks = 7
            # make 7 chunks (with same size) for making 7 tasks
            counts = torch.cumsum(torch.bincount(graph.edata['time']), dim=-1)
            task_ids = (counts / ((graph.num_edges() + 1.) / num_tasks)).long()
            graph.edata['time'] = task_ids[graph.edata['time']]
            # to formulate binary classification problem
            graph.edata['label'] = (graph.edata.pop('label') < 0).long()
        else:
            num_classes = 7 if incr_type == 'domain' else 6
            label_to_class = torch.LongTensor([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 6, 6, 2, 3, 4, 5, 5, 5, 5, 5]) # for balanced split
            graph.edata['label'] = label_to_class[graph.edata.pop('label').squeeze(-1) + 10]
            
        pkl_path = os.path.join(save_path, f'bitcoin_metadata_allIL.pkl')
        # DGL 2.1 overwrites the destination by default.  Re-downloading this
        # immutable split metadata for every stream makes construction depend
        # on network timing instead of the checksummed local dataset cache.
        download(
            'https://github.com/ShinhwanKang/BeGin/raw/main/metadata/'
            'bitcoin_metadata_allIL.pkl',
            pkl_path,
            overwrite=False,
        )
        metadata = pickle.load(open(pkl_path, 'rb'))
        graph.edata['train_mask'] = ((metadata['inner_tvt_split'] % 10) < 8)
        graph.edata['val_mask'] = ((metadata['inner_tvt_split'] % 10) == 8)
        graph.edata['test_mask'] = ((metadata['inner_tvt_split'] % 10) > 8)
        if incr_type == 'domain':
            from gecko.data.splits.common import bitcoin_structural_degree_q4
            from gecko.data.splits.common import canonicalize_logical_edges
            from gecko.data.splits.common import infer_undirected_reverse_arcs

            edge_index = torch.stack(graph.edges(), dim=0)
            undirected = infer_undirected_reverse_arcs(edge_index)
            logical_edges, raw_to_logical, logical_to_raw = canonicalize_logical_edges(
                edge_index, undirected=undirected
            )
            logical_train = torch.zeros(logical_edges.shape[0], dtype=torch.bool)
            for logical_id, raw_ids in logical_to_raw.items():
                raw_train = graph.edata['train_mask'][raw_ids]
                if not torch.equal(raw_train, raw_train[0].expand_as(raw_train)):
                    raise ValueError(
                        f'Reverse Bitcoin arcs disagree on split for logical edge {logical_id}.'
                    )
                logical_train[logical_id] = raw_train[0]
            logical_domains, domain_audit = bitcoin_structural_degree_q4(
                logical_edges,
                logical_train,
                num_nodes=graph.num_nodes(),
            )
            graph.edata['domain'] = logical_domains[raw_to_logical]
            graph.edata['uefa_structural_score'] = domain_audit['scores'][raw_to_logical]
            graph.uefa_domain_audit = domain_audit
        
    else:
        raise NotImplementedError("Tried to load unsupported scenario.")
    
    print("=====CHECK=====")
    print("num_classes:", num_classes, ", num_feats:", num_feats)
    print("graph.edata['train_mask']:", 'train_mask' in graph.edata)
    print("graph.edata['val_mask']:", 'val_mask' in graph.edata)
    print("graph.edata['test_mask']:", 'test_mask' in graph.edata)
    print("graph.edata['label']:", 'label' in graph.edata)
    if incr_type == 'time':
        print("graph.edata['time']:", 'time' in graph.edata)
    if incr_type == 'domain':
        print("graph.edata['domain']:", 'domain' in graph.edata)
    print("===============")
    return num_classes, num_feats, graph




class WikiCSLinkDataset(dgl.data.DGLBuiltinDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=False, transform=None):
        _url = _get_dgl_url('dataset/wiki_cs.zip')
        super(WikiCSLinkDataset, self).__init__(name='wiki_cs',
                                                raw_dir=raw_dir,
                                                url=_url,
                                                force_reload=force_reload,
                                                verbose=verbose)
    def process(self):
        """process raw data to graph, labels and masks"""
        with open(os.path.join(self.raw_path, 'data.json')) as f:
            data = json.load(f)
        features = torch.FloatTensor(np.array(data['features']))
        labels = torch.LongTensor(np.array(data['labels']))

        train_masks = np.array(data['train_masks'], dtype=bool).T
        val_masks = np.array(data['val_masks'], dtype=bool).T
        stopping_masks = np.array(data['stopping_masks'], dtype=bool).T
        test_mask = np.array(data['test_mask'], dtype=bool)

        edges = [[(i, j) for j in js] + [(j, i) for j in js]
                 for i, js in enumerate(data['links'])]
        edges = list(set(chain(*edges)))
        edges = torch.LongTensor([(i, j, j, i) for i, j in edges if i < j]).view(-1, 2)
        g = dgl.graph((edges[:, 0], edges[:, 1]), num_nodes = labels.shape[0])
        g.ndata['feat'] = features
        g.ndata['domain'] = labels
        self._g = g

    def has_cache(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        return os.path.exists(graph_path)

    def save(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        save_graphs(graph_path, self._g)

    def load(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        g, _ = load_graphs(graph_path)
        self._g = g[0]

    @property
    def num_classes(self):
        return 10

    def __len__(self):
        r"""The number of graphs in the dataset."""
        return 1

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self._g


class BitcoinOTCDataset(dgl.data.DGLBuiltinDataset):
    _url = 'https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz'
    _sha1_str = 'c14281f9e252de0bd0b5f1c6e2bae03123938641'

    def __init__(self, dataset_name, raw_dir=None, force_reload=False, verbose=False, transform=None):
        super(BitcoinOTCDataset, self).__init__(name='bitcoinotc',
                                                url=self._url,
                                                raw_dir=raw_dir,
                                                force_reload=force_reload,
                                                verbose=verbose)

    def download(self):
        gz_file_path = os.path.join(self.raw_dir, self.name + '.csv.gz')
        download(self.url, path=gz_file_path)
        if not dgl.data.utils.check_sha1(gz_file_path, self._sha1_str):
            raise UserWarning('File {} is downloaded but the content hash does not match.'
                              'The repo may be outdated or download may be incomplete. '
                              'Otherwise you can create an issue for it.'.format(self.name + '.csv.gz'))
        self._extract_gz(gz_file_path, self.raw_path)

    def process(self):
        filename = os.path.join(self.save_path, '../' + self.name + '.csv')
        data = np.loadtxt(filename, delimiter=',').astype(np.int64)
        data[:, 0:2] = data[:, 0:2] - data[:, 0:2].min()
        delta = datetime.timedelta(days=14).total_seconds()
        time_index = np.around((data[:, 3] - data[:, 3].min()) / delta).astype(np.int64)

        self._graphs = []
        edges = data[:, 0:2]
        rate = data[:, 2]
        # print(data[:, 0:2].min(), data[:, 0:2].max())
        g = dgl.graph((edges[:, 0], edges[:, 1]))
        g.edata['label'] = torch.LongTensor(rate.reshape(-1, 1))
        g.edata['time'] = torch.LongTensor(time_index.reshape(-1))
        g.ndata['feat'] = torch.stack((g.in_degrees(), g.out_degrees()), dim=-1).float()
        self._graphs.append(g)

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

    def __len__(self):
        return len(self.graphs)


    def __getitem__(self, item):
        return self.graphs[item]
    
    @property
    def is_temporal(self):
        return True

    def _extract_gz(self, file, target_dir, overwrite=False):
        if os.path.exists(target_dir) and not overwrite:
            return
        print('Extracting file to {}'.format(target_dir))
        fname = os.path.basename(file)
        makedirs(target_dir)
        out_file_path = os.path.join(target_dir, fname[:-3])
        print(out_file_path)
        with gzip.open(file, 'rb') as f_in:
            with open(out_file_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


class AskUbuntuDataset(dgl.data.DGLBuiltinDataset):
    _url = 'http://snap.stanford.edu/data/sx-askubuntu.txt.gz'
    
    def __init__(self, dataset_name, raw_dir=None, force_reload=False, verbose=False, transform=None):
        super(AskUbuntuDataset, self).__init__(name='askubuntu',
                                                url=self._url,
                                                raw_dir=raw_dir,
                                                force_reload=force_reload,
                                                verbose=verbose)

    def download(self):
        gz_file_path = os.path.join(self.raw_dir, self.name + '.txt.gz')
        download(self.url, path=gz_file_path)
        self._extract_gz(gz_file_path, self.raw_path)


    def process(self):
        filename = os.path.join(self.save_path, '../askubuntu.txt')
        data = np.loadtxt(filename, delimiter=' ').astype(np.int64)
        srcs, dsts, timestamps = data[:, 0].tolist(), data[:, 1].tolist(), data[:, 2].tolist()
        
        edges = set([])
        uniques = []
        interactions = zip(srcs, dsts, timestamps)
        interactions = sorted(interactions, key=lambda x: x[2])
        for s, d, t in interactions:
            if (s, d) not in edges and (d, s) not in edges:
                edges.add((s, d))
                edges.add((d, s))
                uniques.append((s, d, t))
        
        srcs, dsts, timestamps = zip(*uniques)
        months = []
        for t in timestamps:
            dt = datetime.datetime.utcfromtimestamp(t)
            months.append(dt.year * 12 + dt.month)

        months = torch.LongTensor(months)
        srcs, dsts = torch.LongTensor(srcs), torch.LongTensor(dsts)
        months = months - months.min()
        
        chosen_indices = (months >= 18)
        srcs, dsts, months = srcs[chosen_indices], dsts[chosen_indices], months[chosen_indices]
        months = months - months.min()
        print(srcs.shape, months.shape, months.max())
        bi_srcs = torch.stack((srcs, dsts), dim=-1).view(-1)
        bi_dsts = torch.stack((dsts, srcs), dim=-1).view(-1)
        bi_timestamps = torch.repeat_interleave(months, 2, dim=0)
        
        self._graphs = []
        g = dgl.graph((bi_srcs, bi_dsts))
        g.edata['time'] = bi_timestamps
        g.ndata['feat'] = g.in_degrees().float().unsqueeze(-1)
        self._graphs.append(g)

    def has_cache(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        return os.path.exists(graph_path)

    def save(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        save_graphs(graph_path, self._graphs)

    def load(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        self._graphs = load_graphs(graph_path)[0]

    @property
    def graphs(self):
        return self._graphs

    def __len__(self):
        return len(self._graphs)


    def __getitem__(self, item):
        return self._graphs[item]
    
    @property
    def is_temporal(self):
        return True

    def _extract_gz(self, file, target_dir, overwrite=False):
        if os.path.exists(target_dir) and not overwrite:
            return
        print('Extracting file to {}'.format(target_dir))
        fname = os.path.basename(file)
        makedirs(target_dir)
        out_file_path = os.path.join(target_dir, fname[:-3])
        print(out_file_path)
        with gzip.open(file, 'rb') as f_in:
            with open(out_file_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print("DONE")


class FacebookLinkDataset(dgl.data.DGLBuiltinDataset):
    _url = 'http://snap.stanford.edu/data/gemsec_facebook_dataset.tar.gz'
    
    def __init__(self, dataset_name, raw_dir=None, force_reload=False, verbose=False, transform=None):
        super(FacebookLinkDataset, self).__init__(name='facebook',
                                                  url=self._url,
                                                  raw_dir=raw_dir,
                                                  force_reload=force_reload,
                                                  verbose=verbose)

    def download(self):
        gz_file_path = os.path.join(self.raw_dir, self.name + '.tar.gz')
        download(self.url, path=gz_file_path)
        extract_archive(gz_file_path, self.raw_path)

    def process(self):
        domains = ['artist', 'athletes', 'company', 'government', 'new_sites', 'politician', 'public_figure', 'tvshow']
        node_cnt = 0
        all_srcs, all_dsts, all_domains = [], [], []
        # all_neg_edges = []
        for i, domain in enumerate(domains):
            filename = os.path.join(self.save_path, 'facebook_clean_data/' + domain + '_edges.csv')
            y = pd.read_csv(filename)
            srcs, dsts = y['node_1'].values, y['node_2'].values
            num_nodes = max(srcs.max(), dsts.max())
            """
            edges = set(zip(srcs, dsts)).union(set(zip(dsts, srcs)))
            neg_edges = []
            while len(neg_edges) < 25000:
                s, d = np.random.randint(num_nodes), np.random.randint(num_nodes)
                if s >= d: continue
                if (s, d) not in edges:
                    neg_edges.append((s, d))
            all_neg_edges.append(torch.LongTensor(neg_edges) + node_cnt)
            """
            all_srcs.append(torch.LongTensor(srcs) + node_cnt)
            all_dsts.append(torch.LongTensor(dsts) + node_cnt)
            all_domains.append(i * torch.ones(srcs.shape[0], dtype=torch.long))
            node_cnt += num_nodes
        
        all_srcs, all_dsts, all_domains = torch.cat(all_srcs), torch.cat(all_dsts), torch.cat(all_domains)
        bi_srcs = torch.stack((all_srcs, all_dsts), dim=-1).view(-1)
        bi_dsts = torch.stack((all_dsts, all_srcs), dim=-1).view(-1)
        bi_domains = torch.repeat_interleave(all_domains, 2, dim=0)
        self._graphs = []
        graph = dgl.graph((bi_srcs, bi_dsts))
        graph.edata['domain'] = bi_domains
        graph.ndata['feat'] = graph.in_degrees().float().unsqueeze(-1)
        
        """
        all_neg_edges = torch.cat(all_neg_edges, dim=0)
        metadata = {}
        metadata['inner_tvt_splits'] = torch.randperm(all_domains.shape[0]) % 10
        metadata['neg_edges'] = {'val': all_neg_edges[0::2], 'test': all_neg_edges[1::2]}
        pickle.dump(metadata, open('facebook_metadata_domainIL.pkl', 'wb'))
        """
        self._graphs.append(graph)
        
    def has_cache(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        return os.path.exists(graph_path)

    def save(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        save_graphs(graph_path, self._graphs)

    def load(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        self._graphs = load_graphs(graph_path)[0]

    @property
    def graphs(self):
        return self._graphs

    def __len__(self):
        return len(self._graphs)


    def __getitem__(self, item):
        return self._graphs[item]
    
    @property
    def is_temporal(self):
        return True


class MovielensDataset(dgl.data.DGLBuiltinDataset):
    _url = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'

    def __init__(self, dataset_name, raw_dir=None, force_reload=False, verbose=False, transform=None):
        super(MovielensDataset, self).__init__(name='movielens',
                                                 url=self._url,
                                                 raw_dir=raw_dir,
                                                 force_reload=force_reload,
                                                 verbose=verbose)
    def process(self):
        logs = []
        negs = []
        banned = {}
        valid = torch.ones(6040, 3952)
        with open(os.path.join(self.save_path, 'ml-1m/ratings.dat'), 'r') as f:
            for line in f:
                tokens = line.strip().split('::')
                if tokens[2] in ['4', '5']:
                    logs.append((int(tokens[0]), int(tokens[1]), int(tokens[-1])))
                valid[int(tokens[0]) - 1, int(tokens[1]) - 1] = 0

        logs = sorted(logs, key=lambda x: x[-1])
        
        ufeats = np.zeros((6040, 24))
        with open(os.path.join(self.save_path, 'ml-1m/users.dat'), 'r') as f:
            for i, line in enumerate(f):
                tokens = line.strip().split('::')
                if 'F' in tokens[1]:
                    ufeats[i, 0] = 1.
                else:
                    ufeats[i, 1] = 1.
                ufeats[i, 2] = int(tokens[2]) / 56.0
                ufeats[i, 3 + int(tokens[3])] = 1.

        genres =["Action",
                 "Adventure", 
            	"Animation",
            	"Children's",
            	"Comedy",
            	"Crime",
            	"Documentary",
            	"Drama",
            	"Fantasy",
            	"Film-Noir",
            	"Horror",
            	"Musical",
            	"Mystery",
            	"Romance",
            	"Sci-Fi",
            	"Thriller",
            	"War",
            	"Western"]
        gmap = {k: i for i, k in enumerate(genres)}
        ifeats = np.zeros((3952, 18))
        with open(os.path.join(self.save_path, 'ml-1m/movies.dat'), 'r', encoding='latin1') as f:
            for i, line in enumerate(f):
                tokens = line.strip().split('::')
                for mov in tokens[-1].split('|'):
                    ifeats[i, gmap[mov]] = 1.
        
        self.metadata = {}
        self.metadata['edges'] = logs
        self.metadata['feats'] = (ufeats, ifeats)
        self.metadata['time'] = torch.arange(len(logs)) // ((len(logs) // 10) + 1)
        
    def has_cache(self):
        return False

    def save(self):
        pass
        
    def load(self):
        pass
        
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

