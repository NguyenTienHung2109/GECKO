from __future__ import annotations

import torch
import dgl
import copy
from gecko.data.legacy.common import BaseScenarioLoader
from gecko.data.datasets import *
from gecko.data.legacy import evaluator_map

class LPScenarioLoader(BaseScenarioLoader):
    """
        The sceanario loader for link prediction.

        **Usage example:**

            >>> scenario = LPScenarioLoader(dataset_name="ogbl-collab", num_tasks=3, metric="hits@50", 
            ...                             save_path="./data", incr_type="time", task_shuffle=True)

        Bases: ``BaseScenarioLoader``
    """
    def _init_continual_scenario(self):
        from gecko.data.datasets.link import load_linkp_dataset
        self.num_feats, self.__graph, self.__inner_tvt_splits, self.__neg_edges, self.__is_bipartite = load_linkp_dataset(self.dataset_name, self.dataset_load_func, self.incr_type, self.save_path)
        self.num_classes = 1
        
        if self.incr_type in ['class', 'task']:
            # It is impossible to make class-IL and task-IL setting
            raise NotImplementedError
        elif self.incr_type == 'time':
            self.num_tasks = self.__graph.edata['time'].max().item() + 1
            self.__task_ids = self.__graph.edata['time']
            
        elif self.incr_type == 'domain':
            self.num_tasks = self.__graph.edata['domain'].max().item() + 1
            self.__task_ids = self.__graph.edata['domain']
            if self.kwargs is not None and 'task_shuffle' in self.kwargs and self.kwargs['task_shuffle']:
                domain_order = torch.randperm(self.num_tasks)
            else:
                domain_order = torch.arange(self.num_tasks)
            print('domain_order:', domain_order)
            domain_order_inv = torch.arange(self.num_tasks + 1)
            domain_order_inv[domain_order] = torch.arange(self.num_tasks)
            self.__graph.edata['domain'][self.__graph.edata['domain'] < 0] = self.num_tasks
            self.__task_ids = domain_order_inv[self.__graph.edata['domain']]
        
        # set evaluator for the target scenario
        if self.metric is not None:
            if '@' in self.metric:
                metric_name, metric_k = self.metric.split('@')
                self.__evaluator = evaluator_map[metric_name](self.num_tasks, int(metric_k))
            else:
                self.__evaluator = evaluator_map[self.metric](self.num_tasks, self.__task_ids)
        self.__test_results = []
        
    def _update_target_dataset(self):
        # get sources and destinations
        srcs, dsts = self.__graph.edges()
        
        # note that the edges are bi-directed
        is_even = ((torch.arange(self.__inner_tvt_splits.shape[0]) % 2) == 0)
        
        # train/val/test - 8:1:1 random split
        edges_for_train = (self.__inner_tvt_splits < 8)
        if self.incr_type == 'time':
            edges_for_train &= (self.__task_ids <= self._curr_task)
        edges_ready = {'val': ((self.__inner_tvt_splits == 8) & (self.__task_ids == self._curr_task)) & is_even,
                       'test': (self.__inner_tvt_splits > 8) & is_even}
        
        # generate data using only train edges
        target_dataset = dgl.graph((srcs[edges_for_train], dsts[edges_for_train]), num_nodes=self.__graph.num_nodes())
        for k in self.__graph.ndata.keys():
            if (k != 'time' or k != 'domain'): target_dataset.ndata[k] = self.__graph.ndata[k]
        for k in self.__graph.edata.keys():
            if (k != 'time' or k != 'domain'): target_dataset.edata[k] = self.__graph.edata[k][edges_for_train]
            
        # prepare val/test data for current task (containing negative edges)
        target_edges = {_split: torch.stack((srcs[edges_ready[_split]], dsts[edges_ready[_split]]), dim=-1) for _split in ['val', 'test']}
        gt_labels = {_split: torch.cat((self.__task_ids[edges_ready[_split]] + 1,
                             torch.zeros(self.__neg_edges[_split].shape[0], dtype=torch.long)), dim=0) for _split in ['val', 'test']}
        randperms = {_split: torch.randperm(gt_labels[_split].shape[0]) for _split in ['val', 'test']}
        target_edges = {_split: torch.cat((target_edges[_split], self.__neg_edges[_split]), dim=0)[randperms[_split]] for _split in ['val', 'test']}

        # generate train/val/test dataset for current task
        edges_ready['train'] = (edges_for_train & is_even) & (self.__task_ids == self._curr_task)
        target_edges['train'] = torch.stack((srcs[edges_ready['train']], dsts[edges_ready['train']]), dim=-1)
        self.__target_labels = {_split: gt_labels[_split][randperms[_split]] for _split in ['val', 'test']}
        self._target_dataset = {'graph': dgl.add_self_loop(target_dataset),
                                'train': {'edge': target_edges['train']},
                                'val': {'edge': target_edges['val'], 'label': (self.__target_labels['val'] > 0).long()},
                                'test': {'edge': target_edges['test'], 'label': -torch.ones_like(self.__target_labels['test'])}}
        self._target_dataset['train']['label'] = torch.ones(self._target_dataset['train']['edge'].shape[0], dtype=torch.long)
        
    def _update_accumulated_dataset(self):
        # get sources and destinations
        srcs, dsts = self.__graph.edges()
        
        # note that the edges are bi-directed
        is_even = ((torch.arange(self.__inner_tvt_splits.shape[0]) % 2) == 0)
        
        # train/val/test - 8:1:1 random split
        edges_for_train = (self.__inner_tvt_splits < 8)
        if self.incr_type == 'time':
            edges_for_train &= (self.__task_ids <= self._curr_task)
        edges_ready = {'val': ((self.__inner_tvt_splits == 8) & (self.__task_ids <= self._curr_task)) & is_even,
                       'test': (self.__inner_tvt_splits > 8) & is_even}
        target_dataset = dgl.graph((srcs[edges_for_train], dsts[edges_for_train]), num_nodes=self.__graph.num_nodes())
        for k in self.__graph.ndata.keys():
            if (k != 'time' or k != 'domain'): target_dataset.ndata[k] = self.__graph.ndata[k]
        for k in self.__graph.edata.keys():
            if (k != 'time' or k != 'domain'): target_dataset.edata[k] = self.__graph.edata[k][edges_for_train]
            
        # prepare val/test data for current task (containing negative edges)
        target_edges = {_split: torch.stack((srcs[edges_ready[_split]], dsts[edges_ready[_split]]), dim=-1) for _split in ['val']}
        gt_labels = {_split: torch.cat((self.__task_ids[edges_ready[_split]] + 1,
                             torch.zeros(self.__neg_edges[_split].shape[0], dtype=torch.long)), dim=0) for _split in ['val']}

        randperms = {_split: torch.randperm(gt_labels[_split].shape[0]) for _split in ['val']}
        target_edges = {_split: torch.cat((target_edges[_split], self.__neg_edges[_split]), dim=0)[randperms[_split]] for _split in ['val']}
        
        # generate train/val/test dataset for current task
        edges_ready['train'] = (edges_for_train & is_even) & (self.__task_ids <= self._curr_task)
        target_edges['train'] = torch.stack((srcs[edges_ready['train']], dsts[edges_ready['train']]), dim=-1)
        self.__accumulated_labels = {_split: gt_labels[_split][randperms[_split]] for _split in ['val']}
        self.__accumulated_labels['test'] = self.__target_labels['test']
        self._accumulated_dataset = {'graph': dgl.add_self_loop(target_dataset),
                                     'train': {'edge': target_edges['train']},
                                     'val': {'edge': target_edges['val'], 'label': (self.__accumulated_labels['val'] > 0).long()},
                                     'test': self._target_dataset['test']}
        self._accumulated_dataset['train']['label'] = torch.ones(self._accumulated_dataset['train']['edge'].shape[0], dtype=torch.long)
        
    def _get_eval_result_inner(self, preds, target_split):
        """
            The inner function of get_eval_result.
            
            Args:
                preds (torch.Tensor): predicted output of the current model
                target_split (str): target split to measure the performance (spec., 'val' or 'test')
        """
        gt = (self.__target_labels[target_split] > 0).long()
        assert preds.shape == gt.shape, "shape mismatch"
        return self.__evaluator(preds, gt, self.__target_labels[target_split] - 1)
    
    def get_eval_result(self, preds, target_split='test'):
        return self._get_eval_result_inner(preds, target_split)
    
    def get_accum_eval_result(self, preds, target_split='test'):
        """ 
            Compute performance on the accumulated dataset for the given target split.
            It can be used to compute train/val performance during training.
            
            Args:
                preds (torch.Tensor): predicted output of the current model
                target_split (str): target split to measure the performance (spec., 'val' or 'test')
        """
        gt = (self.__accumulated_labels[target_split] > 0).long()
        assert preds.shape == gt.shape, "shape mismatch"
        return self.__evaluator(preds, gt, self.__accumulated_labels[target_split] - 1)
        
    def get_simple_eval_result(self, curr_batch_preds, curr_batch_gts):
        """ 
            Compute performance for the given batch when we ignore task configuration.
            It can be used to compute train/val performance during training.
            
            Args:
                curr_batch_preds (torch.Tensor): predicted output of the current model
                curr_batch_gts (torch.Tensor): ground-truth labels
        """
        return self.__evaluator.simple_eval(curr_batch_preds, curr_batch_gts)
    
    def next_task(self, preds=torch.empty(1)):
        if self.export_mode:
            super().next_task(preds)
        else:
            self.__test_results.append(self._get_eval_result_inner(preds, target_split='test'))
            super().next_task(preds)
            if self._curr_task == self.num_tasks:
                scores = torch.stack(self.__test_results, dim=0)
                scores_np = scores.detach().cpu().numpy()
                ap = scores_np[-1, :-1].mean().item()
                af = (scores_np[np.arange(self.num_tasks), np.arange(self.num_tasks)] - scores_np[-1, :-1]).sum().item() / (self.num_tasks - 1)
                if self.initial_test_result is not None:
                    fwt = (scores_np[np.arange(self.num_tasks-1), np.arange(self.num_tasks-1)+1] - self.initial_test_result.detach().cpu().numpy()[1:-1]).sum() / (self.num_tasks - 1)
                else:
                    fwt = None
                return {'exp_results': scores, 'AP': ap, 'AF': af, 'FWT': fwt}

    def get_current_dataset_for_export(self, _global=False):
        """
            Returns:
                The graph dataset the implemented model uses in the current task
        """
        target_graph = self.__graph if _global else self._target_dataset
        if _global:
            metadata = {'ndata_feat': self.__graph.ndata['feat'], 'task': self.__task_ids}
            metadata['edges'] = self.__graph.edges()
            metadata['neg_edges'] = self.__neg_edges
            metadata['test_edges'] = self._target_dataset['test']['edge']
            metadata['test_labels'] = self.__target_labels['test']
        else:
            metadata = {}
            metadata['edges'] = target_graph['graph'].edges()
            metadata['train_edges'] = target_graph['train']['edge']
            metadata['train_labels'] = target_graph['train']['label']
            metadata['val_edges'] = target_graph['val']['edge']
            metadata['val_labels'] = target_graph['val']['label']
        return metadata

    def export_spec(self):
        """Export LP Domain-IL with globally validated fixed negatives."""
        if self.incr_type != 'domain':
            raise NotImplementedError('UEFA v1 exports binary LP Domain-IL only.')
        from gecko.reproducibility import torch_generator
        from gecko.data.splits.common import canonicalize_logical_edges
        from gecko.data.scenario import build_lp_spec
        from gecko.data.splits.lp import repair_negative_pairs
        from gecko.data.splits.lp import sample_training_negatives

        edge_index = torch.stack(self.__graph.edges(), dim=0)
        positive_pairs, _, logical_to_raw = canonicalize_logical_edges(
            edge_index, undirected=True
        )
        positive_task_ids = torch.empty(positive_pairs.shape[0], dtype=torch.long)
        positive_splits = torch.empty(positive_pairs.shape[0], dtype=torch.long)
        for logical_id, raw_ids in logical_to_raw.items():
            first = int(raw_ids[0])
            if not torch.equal(self.__task_ids[raw_ids], self.__task_ids[first].expand_as(self.__task_ids[raw_ids])):
                raise ValueError(f'Reverse LP arcs disagree on task for logical edge {logical_id}.')
            split_values = self.__inner_tvt_splits[raw_ids]
            if not bool((split_values == split_values[0]).all()):
                raise ValueError(f'Reverse LP arcs disagree on split for logical edge {logical_id}.')
            positive_task_ids[logical_id] = self.__task_ids[first]
            split_value = int(self.__inner_tvt_splits[first])
            positive_splits[logical_id] = 0 if split_value < 8 else (1 if split_value == 8 else 2)
        known_positive_pairs = positive_pairs
        context_positive_pairs = positive_pairs[positive_splits == 0]
        context_edge_index = edge_index[:, self.__inner_tvt_splits < 8]
        valid = positive_task_ids < self.num_tasks
        positive_pairs = positive_pairs[valid]
        positive_task_ids = positive_task_ids[valid]
        positive_splits = positive_splits[valid]
        negatives = {}
        for task_id in range(self.num_tasks):
            train_count = int(((positive_task_ids == task_id) & (positive_splits == 0)).sum())
            train_count = int(
                round(train_count * self.kwargs.get('lp_train_negative_ratio', 1.0))
            )
            negatives[task_id] = {
                'train': sample_training_negatives(
                    num_nodes=self.__graph.num_nodes(),
                    positive_pairs=known_positive_pairs,
                    count=train_count,
                    generator=torch_generator(self.kwargs.get('uefa_seed', 0), 'lp-train-negatives', task_id),
                    undirected=True,
                    bipartite=self.__is_bipartite,
                ),
                'val': repair_negative_pairs(
                    self.__neg_edges['val'].clone(),
                    positive_pairs=known_positive_pairs,
                    count=self.__neg_edges['val'].shape[0],
                    num_nodes=self.__graph.num_nodes(),
                    generator=torch_generator(self.kwargs.get('uefa_seed', 0), 'lp-eval-negatives', task_id, 'val'),
                    undirected=True,
                    bipartite=self.__is_bipartite,
                ),
                'test': repair_negative_pairs(
                    self.__neg_edges['test'].clone(),
                    positive_pairs=known_positive_pairs,
                    count=self.__neg_edges['test'].shape[0],
                    num_nodes=self.__graph.num_nodes(),
                    generator=torch_generator(self.kwargs.get('uefa_seed', 0), 'lp-eval-negatives', task_id, 'test'),
                    undirected=True,
                    bipartite=self.__is_bipartite,
                ),
            }
        return build_lp_spec(
            dataset_name=self.dataset_name,
            metrics=(self.metric,),
            node_features=self.__graph.ndata['feat'],
            positive_pairs=positive_pairs,
            positive_task_ids=positive_task_ids,
            positive_splits=positive_splits,
            negative_pairs_by_task_split=negatives,
            num_tasks=self.num_tasks,
            undirected=True,
            known_positive_pairs=known_positive_pairs,
            context_positive_pairs=context_positive_pairs,
            context_edge_index=context_edge_index,
            bipartite=self.__is_bipartite,
            metadata={'source': 'begin.scenarios.links.LPScenarioLoader'},
        )


class LCScenarioLoader(BaseScenarioLoader):
    """
        The sceanario loader for link classification.

        **Usage example:**

            >>> scenario = LCScenarioLoader(dataset_name="bitcoin", num_tasks=3, metric="accuracy", 
            ...                             save_path="./data", incr_type="task", task_shuffle=True)
            
            >>> scenario = LCScenarioLoader(dataset_name="bitcoin", num_tasks=7, metric="aucroc", 
            ...                             save_path="./data", incr_type="time")

        Bases: ``BaseScenarioLoader``
    """
    def _init_continual_scenario(self):
        from gecko.data.datasets.link import load_linkc_dataset
        self.num_classes, self.num_feats, self.__graph = load_linkc_dataset(self.dataset_name, self.dataset_load_func, self.incr_type, self.save_path)
        if 'domain' in self.__graph.edata: self.__domain_info = self.__graph.edata['domain']
        if 'time' in self.__graph.edata: self.__time_splits = self.__graph.edata['time']
        
        if self.incr_type in ['domain']:
            self.num_tasks = self.__domain_info.max().item() + 1
            if self.kwargs is not None and self.kwargs.get('task_shuffle', False):
                domain_order = torch.randperm(self.num_tasks)
            else:
                domain_order = torch.arange(self.num_tasks)
            domain_order_inverse = torch.empty_like(domain_order)
            domain_order_inverse[domain_order] = torch.arange(self.num_tasks)
            self.__task_ids = domain_order_inverse[self.__domain_info]
            print('domain_order:', domain_order)
        elif self.incr_type == 'time':
            # split into tasks using timestamp
            self.num_tasks = self.__time_splits.max().item() + 1
            print('num_tasks:', self.num_tasks)
            self.__task_ids = self.__time_splits
        elif self.incr_type in ['class', 'task']:
            # determine task configuration
            if self.kwargs is not None and 'task_orders' in self.kwargs:
                self.__splits = tuple([torch.LongTensor(class_ids) for class_ids in self.kwargs['task_orders']])
            elif self.kwargs is not None and 'task_shuffle' in self.kwargs and self.kwargs['task_shuffle']:
                self.__splits = torch.split(torch.randperm(self.num_classes), self.num_classes // self.num_tasks)[:self.num_tasks]
            else:
                self.__splits = torch.split(torch.arange(self.num_classes), self.num_classes // self.num_tasks)[:self.num_tasks]
            
            print('class split information:', self.__splits)
            # compute task ids for each instance and remove time information (since it is unnecessary)
            id_to_task = self.num_tasks * torch.ones(self.__graph.edata['label'].max() + 1).long()
            for i in range(self.num_tasks):
                id_to_task[self.__splits[i]] = i
            self.__task_ids = id_to_task[self.__graph.edata['label']]
            # ignore classes which are not used in the tasks
            self.__graph.edata['test_mask'] = self.__graph.edata['test_mask'] & (self.__task_ids < self.num_tasks)
            
        # we need to provide task information (only for task-IL)
        if self.incr_type == 'task':
            self.__task_masks = torch.zeros(self.num_tasks + 1, self.num_classes).bool()
            for i in range(self.num_tasks):
                self.__task_masks[i, self.__splits[i]] = True
        
        # set evaluator for the target scenario
        if self.metric is not None:
            self.__evaluator = evaluator_map[self.metric](self.num_tasks, self.__task_ids)
        self.__test_results = []
        
    def _update_target_dataset(self):
        target_dataset = copy.deepcopy(self.__graph)
        
        # set train/val/test split for the current task
        target_dataset.edata['train_mask'] = self.__graph.edata['train_mask'] & (self.__task_ids == self._curr_task)
        target_dataset.edata['val_mask'] = self.__graph.edata['val_mask'] & (self.__task_ids == self._curr_task)
        target_dataset.edata['test_mask'] = self.__graph.edata['test_mask']
        
        # hide labels of test nodes
        target_dataset.edata['label'] = self.__graph.edata['label'].clone()
        target_dataset.edata['label'][target_dataset.edata['test_mask'] | (self.__task_ids != self._curr_task)] = -1
        
        if self.incr_type == 'class':
            # for class-IL, no need to change
            self._target_dataset = target_dataset
        elif self.incr_type == 'task':
            # for task-IL, we need task information. BeGin provide the information with 'task_specific_mask'
            self._target_dataset = target_dataset
            self._target_dataset.edata['task_specific_mask'] = self.__task_masks[self.__task_ids]
        elif self.incr_type == 'time':
            # for time-IL, we need to hide unseen nodes and information at the current timestamp
            srcs, dsts = target_dataset.edges()
            edges_ready = (self.__task_ids <= self._curr_task)
            self._target_dataset = dgl.graph((srcs[edges_ready], dsts[edges_ready]), num_nodes=self.__graph.num_nodes())
            for k in target_dataset.ndata.keys():
                self._target_dataset.ndata[k] = target_dataset.ndata[k]
            for k in target_dataset.edata.keys():
                self._target_dataset.edata[k] = target_dataset.edata[k][edges_ready]
        elif self.incr_type == 'domain':
            self._target_dataset = target_dataset
            for key in ('domain', 'uefa_structural_score'):
                if key in self._target_dataset.edata:
                    self._target_dataset.edata.pop(key)
        
    def _update_accumulated_dataset(self):
        target_dataset = copy.deepcopy(self.__graph)
        
        # set train/val/test split for the current task
        target_dataset.edata['train_mask'] = self.__graph.edata['train_mask'] & (self.__task_ids <= self._curr_task)
        target_dataset.edata['val_mask'] = self.__graph.edata['val_mask'] & (self.__task_ids <= self._curr_task)
        target_dataset.edata['test_mask'] = self.__graph.edata['test_mask']
        
        # hide labels of test nodes
        target_dataset.edata['label'] = self.__graph.edata['label'].clone()
        target_dataset.edata['label'][target_dataset.edata['test_mask'] | (self.__task_ids > self._curr_task)] = -1
        
        if self.incr_type == 'class':
            # for class-IL, no need to change
            self._accumulated_dataset = target_dataset
        elif self.incr_type == 'task':
            # for task-IL, we need task information. BeGin provide the information with 'task_specific_mask'
            self._accumulated_dataset = target_dataset
            self._accumulated_dataset.edata['task_specific_mask'] = self.__task_masks[self.__task_ids]
        elif self.incr_type == 'time':
            # for time-IL, we need to hide unseen nodes and information at the current timestamp
            srcs, dsts = target_dataset.edges()
            edges_ready = (self.__task_ids <= self._curr_task)
            self._accumulated_dataset = dgl.graph((srcs[edges_ready], dsts[edges_ready]), num_nodes=self.__graph.num_nodes())
            for k in target_dataset.ndata.keys():
                self._accumulated_dataset.ndata[k] = target_dataset.ndata[k]
            for k in target_dataset.edata.keys():
                self._accumulated_dataset.edata[k] = target_dataset.edata[k][edges_ready]
        elif self.incr_type == 'domain':
            self._accumulated_dataset = target_dataset
            for key in ('domain', 'uefa_structural_score'):
                if key in self._accumulated_dataset.edata:
                    self._accumulated_dataset.edata.pop(key)
            
    def _get_eval_result_inner(self, preds, target_split):
        """
            The inner function of get_eval_result.
            
            Args:
                preds (torch.Tensor): predicted output of the current model
                target_split (str): target split to measure the performance (spec., 'val' or 'test')
        """
        if self.incr_type == 'time':
            # for Time-IL we evaluate the performance only with currently seen nodes
            gt = self.__graph.edata['label'][self.__task_ids <= self._curr_task][self._target_dataset.edata[target_split + '_mask']]
            assert preds.shape == gt.shape, "shape mismatch"
            return self.__evaluator(preds, gt, torch.arange(self.__graph.num_edges())[self.__task_ids <= self._curr_task][self._target_dataset.edata[target_split + '_mask']])
        else:
            gt = self.__graph.edata['label'][self._target_dataset.edata[target_split + '_mask']]
            assert preds.shape == gt.shape, "shape mismatch"
            return self.__evaluator(preds, gt, torch.arange(self._target_dataset.num_edges())[self._target_dataset.edata[target_split + '_mask']])
    
    def get_eval_result(self, preds, target_split='test'):
        return self._get_eval_result_inner(preds, target_split)
    
    def get_accum_eval_result(self, preds, target_split='test'):
        """ 
            Compute performance on the accumulated dataset for the given target split.
            It can be used to compute train/val performance during training.
            
            Args:
                preds (torch.Tensor): predicted output of the current model
                target_split (str): target split to measure the performance (spec., 'val' or 'test')
        """
        if self.incr_type == 'time':
            # for Time-IL we evaluate the performance only with currently seen nodes
            gt = self.__graph.edata['label'][self.__task_ids <= self._curr_task][self._accumulated_dataset.edata[target_split + '_mask']]
            assert preds.shape == gt.shape, "shape mismatch"
            return self.__evaluator(preds, gt, torch.arange(self.__graph.num_edges())[self.__task_ids <= self._curr_task][self._accumulated_dataset.edata[target_split + '_mask']])
        else:
            gt = self.__graph.edata['label'][self._accumulated_dataset.edata[target_split + '_mask']]
            assert preds.shape == gt.shape, "shape mismatch"
            return self.__evaluator(preds, gt, torch.arange(self._target_dataset.num_edges())[self._accumulated_dataset.edata[target_split + '_mask']])
        
    def get_simple_eval_result(self, curr_batch_preds, curr_batch_gts):
        """ 
            Compute performance for the given batch when we ignore task configuration.
            It can be used to compute train/val performance during training.
            
            Args:
                curr_batch_preds (torch.Tensor): predicted output of the current model
                curr_batch_gts (torch.Tensor): ground-truth labels
        """
        return self.__evaluator.simple_eval(curr_batch_preds, curr_batch_gts)
    
    def next_task(self, preds=torch.empty(1)):
        if self.export_mode:
            super().next_task(preds)
        else:
            self.__test_results.append(self._get_eval_result_inner(preds, target_split='test'))
            super().next_task(preds)
            if self._curr_task == self.num_tasks:
                scores = torch.stack(self.__test_results, dim=0)
                scores_np = scores.detach().cpu().numpy()
                ap = scores_np[-1, :-1].mean().item()
                af = (scores_np[np.arange(self.num_tasks), np.arange(self.num_tasks)] - scores_np[-1, :-1]).sum().item() / (self.num_tasks - 1)
                if self.initial_test_result is not None:
                    fwt = (scores_np[np.arange(self.num_tasks-1), np.arange(self.num_tasks-1)+1] - self.initial_test_result.detach().cpu().numpy()[1:-1]).sum() / (self.num_tasks - 1)
                else:
                    fwt = None
                return {'exp_results': scores, 'AP': ap, 'AF': af, 'FWT': fwt}
    
    def get_current_dataset_for_export(self, _global=False):
        """
            Returns:
                The graph dataset the implemented model uses in the current task
        """
        target_graph = self.__graph if _global else self._target_dataset
        metadata = {'num_classes': self.num_classes, 'ndata_feat': self.__graph.ndata['feat'], 'task': self.__task_ids} if _global else {}
        if _global and self.incr_type == 'task':  metadata['task_specific_mask'] = self.__task_masks[self.__task_ids]
        metadata['edges'] = target_graph.edges()
        metadata['train_mask'] = target_graph.edata['train_mask']
        metadata['val_mask'] = target_graph.edata['val_mask']
        if _global: metadata['test_mask'] = target_graph.edata['test_mask']
        metadata['label'] = copy.deepcopy(target_graph.edata['label'])
        return metadata

    def export_spec(self):
        """Export an immutable global-task LC specification without advancing."""
        from gecko.data.splits.common import infer_undirected_reverse_arcs
        from gecko.data.scenario import build_lc_spec

        edge_index = torch.stack(self.__graph.edges(), dim=0)
        task_class_sets = {}
        if self.incr_type in ['task', 'class']:
            task_class_sets = {
                task_id: class_ids.clone()
                for task_id, class_ids in enumerate(self.__splits)
            }
        metadata = {'source': 'begin.scenarios.links.LCScenarioLoader'}
        if self.incr_type == 'domain':
            metadata.update({
                'constructor': 'bitcoin_structural_degree_q4' if self.dataset_name == 'bitcoin' else 'custom',
                'constructor_version': 1,
            })
            metadata.update(getattr(self.__graph, 'uefa_domain_audit', {}))
        return build_lc_spec(
            dataset_name=self.dataset_name,
            incremental_type=self.incr_type,
            metrics=(self.metric, 'macro_f1'),
            edge_index=edge_index,
            node_features=self.__graph.ndata['feat'],
            raw_labels=self.__graph.edata['label'].squeeze(),
            raw_task_ids=self.__task_ids,
            raw_train_mask=self.__graph.edata['train_mask'],
            raw_validation_mask=self.__graph.edata['val_mask'],
            raw_test_mask=self.__graph.edata['test_mask'],
            num_tasks=self.num_tasks,
            num_classes=self.num_classes,
            undirected=infer_undirected_reverse_arcs(edge_index),
            task_class_sets=task_class_sets,
            domains=self.__task_ids if self.incr_type == 'domain' else None,
            metadata=metadata,
        )




_RELOCATED_EXPORTS = {'load_linkc_dataset': ('gecko.data.datasets.link', 'load_linkc_dataset'), 'load_linkp_dataset': ('gecko.data.datasets.link', 'load_linkp_dataset')}

def __getattr__(name: str):
    from importlib import import_module
    if name not in _RELOCATED_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _RELOCATED_EXPORTS[name]
    return getattr(import_module(module), symbol)
