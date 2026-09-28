from __future__ import annotations

import torch
import dgl
import copy
from gecko.data.legacy.common import BaseScenarioLoader
from gecko.data.datasets import *
from gecko.data.legacy import evaluator_map

class NCScenarioLoader(BaseScenarioLoader):
    """
        The sceanario loader for node classification problems.

        **Usage example:**

            >>> scenario = NCScenarioLoader(dataset_name dataset_object=None, num_tasks=3, metric="accuracy", 
            ...                             save_path="./data", incr_type="task", task_shuffle=True)

        Bases: ``BaseScenarioLoader``
    """
    
    def _init_continual_scenario(self):
        from gecko.data.datasets.node import load_node_dataset
        self.num_classes, self.num_feats, self.__graph, self.__cover_rule = load_node_dataset(self.dataset_name, self.dataset_load_func, self.incr_type, self.save_path)
        if self.incr_type in ['class', 'task']:
            # determine task configuration
            if self.kwargs is not None and 'task_orders' in self.kwargs:
                self.__splits = tuple([torch.LongTensor(class_ids) for class_ids in self.kwargs['task_orders']])
            elif self.kwargs is not None and 'task_shuffle' in self.kwargs and self.kwargs['task_shuffle']:
                self.__splits = torch.split(torch.randperm(self.num_classes), self.num_classes // self.num_tasks)[:self.num_tasks]
            else:
                self.__splits = torch.split(torch.arange(self.num_classes), self.num_classes // self.num_tasks)[:self.num_tasks]
            
            print('class split information:', self.__splits)
            # compute task ids for each node
            id_to_task = self.num_tasks * torch.ones(self.__graph.ndata['label'].max() + 1).long()
            for i in range(self.num_tasks):
                id_to_task[self.__splits[i]] = i
            self.__task_ids = id_to_task[self.__graph.ndata['label']]
            
            # ignore classes which are not used in the tasks
            self.__graph.ndata['test_mask'] = self.__graph.ndata['test_mask'] & (self.__task_ids < self.num_tasks)
        elif self.incr_type == 'time':
            # compute task ids for each node
            self.__task_ids = self.__graph.ndata['time']
            if self.num_tasks != self.__task_ids.max().item() + 1:
                print("WARNING: Mismatch between the number of tasks and the processed data. Please check again.")
            # overwrite num_tasks
            self.num_tasks = self.__task_ids.max().item() + 1
        elif self.incr_type == 'domain':
            # num_tasks only depends on the number of domains
            self.num_tasks = self.__graph.ndata['domain'].max().item() + 1
            # determine task configuration
            if self.kwargs is not None and 'task_shuffle' in self.kwargs and self.kwargs['task_shuffle']:
                self.__task_order = torch.randperm(self.num_tasks)
                print('domain_order:', self.__task_order)
                self.__task_ids = self.__task_order[self.__graph.ndata['domain']]
            else:
                self.__task_ids = self.__graph.ndata['domain']
                
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
        target_dataset = self.__graph.clone()
        
        # conceal unnecessary information
        for k, v in self.__cover_rule.items():
            if v == 'node': target_dataset.ndata.pop(k)
            elif v == 'edge': target_dataset.edata.pop(k)
        target_dataset.ndata['feat'] = self.__graph.ndata['feat'].clone()
        target_dataset.ndata['label'] = self.__graph.ndata['label'].clone()
        target_dataset.ndata['train_mask'] = self.__graph.ndata['train_mask'].clone()
        target_dataset.ndata['val_mask'] = self.__graph.ndata['val_mask'].clone()
        target_dataset.ndata['test_mask'] = self.__graph.ndata['test_mask'].clone()
        
        # update train/val/test mask for the current task
        target_dataset.ndata['train_mask'] = target_dataset.ndata['train_mask'] & (self.__task_ids == self._curr_task)
        target_dataset.ndata['val_mask'] = target_dataset.ndata['val_mask'] & (self.__task_ids == self._curr_task)
        target_dataset.ndata['label'][target_dataset.ndata['test_mask'] | (self.__task_ids > self._curr_task)] = -1
        
        if self.incr_type == 'class':
            # for class-IL, no need to change
            self._target_dataset = target_dataset
        elif self.incr_type == 'task':
            # for task-IL, we need task information. BeGin provide the information with 'task_specific_mask'
            self._target_dataset = target_dataset
            self._target_dataset.ndata['task_specific_mask'] = self.__task_masks[self.__task_ids]
        elif self.incr_type == 'time':
            # for time-IL, we need to hide unseen nodes and information at the current timestamp
            
            # remain only seen nodes and edges
            srcs, dsts = target_dataset.edges()
            nodes_ready = self.__task_ids <= self._curr_task
            edges_ready = (self.__task_ids[srcs] <= self._curr_task) & (self.__task_ids[dsts] <= self._curr_task)
            self._target_dataset = dgl.graph((srcs[edges_ready], dsts[edges_ready]), num_nodes=self.__graph.num_nodes())
            
            # cover the information of the unseen nodes/edges
            for k in target_dataset.ndata.keys():
                self._target_dataset.ndata[k] = target_dataset.ndata[k]
                if self._target_dataset.ndata[k].dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
                    self._target_dataset.ndata[k][~nodes_ready] = -1
                else:
                    self._target_dataset.ndata[k][~nodes_ready] = 0
            for k in target_dataset.edata.keys():
                self._target_dataset.edata[k] = target_dataset.edata[k][edges_ready]
            
            # update test mask (exclude unseen test nodes)
            self._target_dataset.ndata['test_mask'] = self._target_dataset.ndata['test_mask'] & (self.__task_ids <= self._curr_task)
        elif self.incr_type == 'domain':
            # for domain-IL, no need to change
            self._target_dataset = target_dataset
            
    def _update_accumulated_dataset(self):
        target_dataset = self.__graph.clone()
        
        # conceal unnecessary information
        for k, v in self.__cover_rule.items():
            if v == 'node': target_dataset.ndata.pop(k)
            elif v == 'edge': target_dataset.edata.pop(k)

        target_dataset.ndata['feat'] = self.__graph.ndata['feat'].clone()
        target_dataset.ndata['label'] = self.__graph.ndata['label'].clone()
        target_dataset.ndata['train_mask'] = self.__graph.ndata['train_mask'].clone()
        target_dataset.ndata['val_mask'] = self.__graph.ndata['val_mask'].clone()
        target_dataset.ndata['test_mask'] = self.__graph.ndata['test_mask'].clone()
        
        # update train/val/test mask for the current task
        target_dataset.ndata['train_mask'] = target_dataset.ndata['train_mask'] & (self.__task_ids <= self._curr_task)
        target_dataset.ndata['val_mask'] = target_dataset.ndata['val_mask'] & (self.__task_ids <= self._curr_task)
        target_dataset.ndata['label'][target_dataset.ndata['test_mask'] | (self.__task_ids > self._curr_task)] = -1
        
        if self.incr_type == 'class':
            # for class-IL, no need to change
            self._accumulated_dataset = target_dataset
        elif self.incr_type == 'task':
            # for task-IL, we need task information. BeGin provide the information with 'task_specific_mask'
            self._accumulated_dataset = target_dataset
            self._accumulated_dataset.ndata['task_specific_mask'] = self.__task_masks[self.__task_ids]
        elif self.incr_type == 'time':
            # for time-IL, we need to hide unseen nodes and information at the current timestamp
            srcs, dsts = target_dataset.edges()
            nodes_ready = self.__task_ids <= self._curr_task
            edges_ready = (self.__task_ids[srcs] <= self._curr_task) & (self.__task_ids[dsts] <= self._curr_task)
            self._accumulated_dataset = dgl.graph((srcs[edges_ready], dsts[edges_ready]), num_nodes=self.__graph.num_nodes())
            
            # cover the information of the unseen nodes/edges
            for k in target_dataset.ndata.keys():
                self._accumulated_dataset.ndata[k] = target_dataset.ndata[k]
                if self._accumulated_dataset.ndata[k].dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
                    self._accumulated_dataset.ndata[k][~nodes_ready] = -1
                else:
                    self._accumulated_dataset.ndata[k][~nodes_ready] = 0
            for k in target_dataset.edata.keys():
                self._accumulated_dataset.edata[k] = target_dataset.edata[k][edges_ready]
                
            # update test mask (exclude unseen test nodes)
            self._accumulated_dataset.ndata['test_mask'] = self._accumulated_dataset.ndata['test_mask'] & (self.__task_ids <= self._curr_task)
        elif self.incr_type == 'domain':
            self._accumulated_dataset = target_dataset
            
    def _get_eval_result_inner(self, preds, target_split):
        """
            The inner function of get_eval_result.
            
            Args:
                preds (torch.Tensor): predicted output of the current model
                target_split (str): target split to measure the performance (spec., 'val' or 'test')
        """
        gt = self.__graph.ndata['label'][self._target_dataset.ndata[target_split + '_mask']]
        assert preds.shape == gt.shape, "shape mismatch"
        return self.__evaluator(preds, gt, torch.arange(self._target_dataset.num_nodes())[self._target_dataset.ndata[target_split + '_mask']])
    
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
        
        gt = self.__graph.ndata['label'][self._accumulated_dataset.ndata[target_split + '_mask']]
        assert preds.shape == gt.shape, "shape mismatch"
        return self.__evaluator(preds, gt, torch.arange(self._accumulated_dataset.num_nodes())[self._accumulated_dataset.ndata[target_split + '_mask']])
    
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
        metadata['train_mask'] = target_graph.ndata['train_mask']
        metadata['val_mask'] = target_graph.ndata['val_mask']
        if _global: metadata['test_mask'] = target_graph.ndata['test_mask']
        metadata['label'] = copy.deepcopy(target_graph.ndata['label'])
        return metadata

    def export_spec(self):
        """Export an immutable global-task NC specification without advancing."""
        from gecko.data.scenario import build_nc_spec

        task_class_sets = {}
        if self.incr_type in ['task', 'class']:
            task_class_sets = {
                task_id: class_ids.clone()
                for task_id, class_ids in enumerate(self.__splits)
            }
        domains = None
        if self.incr_type == 'domain':
            domains = self.__graph.ndata['domain'].clone()
        metadata = {
            'source': 'begin.scenarios.nodes.NCScenarioLoader',
            'target_type': (
                'multi_label' if self.__graph.ndata['label'].ndim > 1
                else 'single_label'
            ),
        }
        if hasattr(self.__graph, 'uefa_context_edge_features'):
            metadata.update({
                'feature_provenance': 'strict_local_derived',
                'feature_derivation': 'mean_outgoing_visible_edge_features',
                'context_edge_features': self.__graph.uefa_context_edge_features,
                'context_edge_feature_valid_mask': (
                    self.__graph.uefa_context_edge_feature_valid_mask
                ),
            })
        edge_index = torch.stack(self.__graph.edges(), dim=0)
        return build_nc_spec(
            dataset_name=self.dataset_name,
            incremental_type=self.incr_type,
            metrics=(self.metric,),
            edge_index=edge_index,
            node_features=self.__graph.ndata['feat'],
            labels=self.__graph.ndata['label'],
            task_ids=self.__task_ids,
            train_mask=self.__graph.ndata['train_mask'],
            validation_mask=self.__graph.ndata['val_mask'],
            test_mask=self.__graph.ndata['test_mask'],
            num_tasks=self.num_tasks,
            num_classes=self.num_classes,
            task_class_sets=task_class_sets,
            domains=domains,
            metadata=metadata,
        )




_RELOCATED_EXPORTS = {'load_node_dataset': ('gecko.data.datasets.node', 'load_node_dataset')}

def __getattr__(name: str):
    from importlib import import_module
    if name not in _RELOCATED_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _RELOCATED_EXPORTS[name]
    return getattr(import_module(module), symbol)
