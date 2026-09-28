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


_RELOCATED_DATASETS = {'DGLGNNBenchmarkDataset': 'graph', 'NYCTaxiDataset': 'graph', 'DglGraphPropPredDatasetWithTaskMask': 'graph', 'WikiCSLinkDataset': 'link', 'BitcoinOTCDataset': 'link', 'AromaticityDataset': 'graph', 'TwitchGamerNodeDataset': 'node', 'OgbgPpaSampledDataset': 'graph', 'AskUbuntuDataset': 'link', 'FacebookLinkDataset': 'link', 'SentimentGraphDataset': 'graph', 'ZINCGraphDataset': 'graph', 'AQSOLGraphDataset': 'graph', 'MovielensDataset': 'link'}

def __getattr__(name: str):
    from importlib import import_module
    if name not in _RELOCATED_DATASETS:
        raise AttributeError(name)
    return getattr(import_module("gecko.data.datasets." + _RELOCATED_DATASETS[name]), name)
