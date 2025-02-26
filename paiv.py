import argparse
# from collections import deque
from abc import ABC, abstractmethod
import math
from mimetypes import init

from re import A
from typing import List
import torch
from torch import nn

from torch.nn import Module
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchmetrics import ConfusionMatrix
from graph.sai_graph_generator import SAIGraph
from training_strategies.local_update_strategy import StandardLocalUpdateV1_hessian,IdealDistillationLocalUpdate, StandardLocalUpdate, DistillationLocalUpdate, SelfDistillationLocalUpdate, StandardLocalUpdateV1, VirtualDistillationLocalUpdate, CFAGE_LocalModelUpdate
from training_strategies.aggregation_strategy import FedAvg, SocialFedAvg, SocialLocalGlobalDiffUpdate, consensus_federated_average,SocialLocalGlobalDiffUpdate_hessian_diag
import numpy as np
#from utils.messaging import Message
from message import Message
import copy

from queue import PriorityQueue

import os


class AbstractPaiv(ABC):
    def __init__(self, id: int, args: argparse.Namespace, graph: SAIGraph, dataset: Dataset, data_idxs: List, model: Module):
        # PAIV id
        self._id = id
        # input args
        self._args = args
        # self trust
        # FIXME: setting the self trust to 1 makes sense only when all weights are 1. Otherwise a smarter way should be found.
        self._selfconfidence = 1
        # graph object
        self._graph = graph
        # list of idxs of the dataset
        self._dataset = dataset
        self._data_idxs = data_idxs
        # local model
        self._model = model
        # buffer to collect other models
        self._msg_buffer = None

    @property
    def id(self):
        return self._id

    @property
    def args(self):
        return self._args

    @property
    def selfconfidence(self):
        return self._selfconfidence

    @property
    def graph(self):
        return self._graph

    @property
    def dataset(self):
        return self._dataset

    @property
    def data_idxs(self):
        return self._data_idxs

    @property
    def model(self):
        return self._model

    @property
    def msg_buffer(self):
        return self._msg_buffer

    @property
    def best_valid_loss(self):
        return self._best_valid_loss

    @property
    def train_role(self):
        return self._train_role

    @id.setter
    def id(self, value):
        self._id = value

    @selfconfidence.setter
    def selfconfidence(self, value):
        self._selfconfidence = value

    @graph.setter
    def graph(self, value):
        self._graph = value

    @dataset.setter
    def dataset(self, value):
        self._dataset = value

    @data_idxs.setter
    def data_idxs(self, value):
        self._data_idxs = value

    @model.setter
    def model(self, value):
        self._model = value

    @msg_buffer.setter
    def msg_buffer(self, value):
        self._msg_buffer = value

    @selfconfidence.setter
    def selfconfidence(self, value):
        self._selfconfidence = value

    @best_valid_loss.setter
    def best_valid_loss(self, value):
        self._best_valid_loss = value

    @train_role.setter
    def train_role(self, value):
        self._train_role = value

    def receive(self, message):
        self._msg_buffer.append(message)

    def clear_msg_buffer(self):
        self._msg_buffer.clear()


class simplePaiv(AbstractPaiv):

    def __init__(self, id, args, graph, dataset, data_idxs, model, train_role):
        super().__init__(id=id, args=args, graph=graph, dataset=dataset, data_idxs=data_idxs,
                         model=model)
        self._msg_buffer = PriorityQueue()

        # training related properties
        self._loss_local = []
        # self._valid_loss = np.Inf

        self._err_func = nn.CrossEntropyLoss(reduction='none')
        self._val_err_func = nn.CrossEntropyLoss()

        # train_size = int(np.floor(len(data_idxs) * (1-self.args.val_split)))
        # randidx = torch.randperm(len(data_idxs))
        # self.train_idx = data_idxs[randidx[:train_size]]
        # self.val_idx = data_idxs[randidx[train_size:]]

        self.train_idx = data_idxs['train']
        self.val_idx = data_idxs['validation']
        self.train_size = len(self.train_idx)
        print(
            f'Data distrib: {torch.unique(dataset.targets[self.val_idx],return_counts=True)}')

        self._local_strategy = StandardLocalUpdateV1(
            self._args, dataset=self._dataset, idxs=(self.train_idx, self.val_idx), loss_func=self._err_func,val_loss_func=self._val_err_func)

        self._best_valid_loss = np.Inf

        self._train_role = train_role

        self.aggregation_func = self._args.aggregation_func

        self.toggle_aggregation = self.args.toggle_aggregate_first

    def train(self, current_time):
        # here we implement the atomic training that includes interaction between local and peers' models
        # workflow:
        # 1) process peers' models
        # 2) train local model embedding peers' knowledge
        if self.args.verbose:
            print("PAIV", self._id, "/ The msg_buffer size is:",
                  len(self._msg_buffer.queue))
        
        # DEBUG: we print the models of selected nodes
        if self._args.toggle_output_model and self._id in [0,1]:
            filename = f'stats/{self._args.outfolder}/models/model_{self._id}_at_{current_time}_{self._args.seed}.pt'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            print(f'Saving models in {filename}')
            torch.save(self._model, filename)

        active_neighs = list()
        if not self._msg_buffer.empty():
            # implement logic for selecting the models to aggregate
            # simple: use all models in the buffer
            models = {}
            neigh_data_size = {}
            # note: the [:] is necessary, it creates a copy of
            for msg in self._msg_buffer.queue[:]:
                # self._msg_buffer.queue and the removal with the .get() is applied
                # only when msg.time < current_time
                if msg.time < current_time:
                    if self.args.verbose:
                        print("TRAIN: adding model from", msg.source,
                              "generated at time", msg.time)
                    # associate model with their source node for weighting
                    models[msg.source] = msg.model.state_dict()
                    neigh_data_size[msg.source] = msg.dataset_size
                    active_neighs.append(msg.source)
                    self._msg_buffer.get()  # remove the processed message from the message buffer
            if self.args.verbose:
                print("-- ", self._msg_buffer.qsize(),
                      "messages already available for the next training epoch")
        
        # print(f'Node {self._id} is aggregating the models coming from {active_neighs}')

        # save the current model
        prev_model = copy.deepcopy(self._model)
        # perform merging if you have received messages from your neighbors
        if active_neighs:
            if self._train_role == 'standalone':
                if self.aggregation_func == 'fed_avg':
                    # aggregate models using social federated average aggregation
                    self._social_aggregation(
                        active_neighs=active_neighs, models=models, data_sizes=neigh_data_size)
                elif self.aggregation_func == 'fed_diff':
                    self._social_diff_aggregation(
                        active_neighs=active_neighs, models=models, data_sizes=neigh_data_size)
                elif self.aggregation_func == 'cfa':
                    self._consensus_fa(active_neighs=active_neighs, models=models,data_sizes=neigh_data_size)
            elif self._train_role == 'fed_client':
                # simply substitute local with global model
                self._substitute_local_with_global_model(models=models)
        
        # DEBUG: we print the models of selected nodes
        if self._args.toggle_output_model and self._id in [0,1]:
            filename = f'stats/{self._args.outfolder}/models/model_after_feddiff_{self._id}_at_{current_time}_{self._args.seed}.pt'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            print(f'Saving models in {filename}')
            torch.save(self._model, filename)

        # train local model
        w, loss, vloss = self._local_strategy.train(
            net=self._model.to(self._args.device))

        # update the local model if the new validation loss improves over the previous one
        # Note: this works only if the validation set is reliable. Not sure what happens in highly non-iid data distribution. 
        always_update = True
        if always_update or vloss < self._best_valid_loss:
            self._best_valid_loss = vloss
            self._model.load_state_dict(w)
        else:
            # revert to the model before the aggregation + retrain
            print("No improvement in valid loss, reverting to backup copy of the model")
            self._model.load_state_dict(prev_model.state_dict())

        # DEBUG: we print the models of selected nodes
        if self._args.toggle_output_model and self._id in [0,1]:
            filename = f'stats/{self._args.outfolder}/models/model_after_retrain_{self._id}_at_{current_time}_{self._args.seed}.pt'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            print(f'Saving models in {filename}')
            torch.save(self._model, filename)

        
        # return the loss after the training epoch 
        return loss




    def validate(self):
        return self._best_valid_loss

    def test(self, dataset, model=None):
        if model is None:
            model = self._model

        # TODO: here we apply the local model to the validation set
        batch_valid_loss = 0
        batch_accuracy = 0
        ldr_dataset = DataLoader(
            dataset, batch_size=self.args.local_bs)
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(ldr_dataset):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)

                log_probs = model(images)

                y_prob = nn.Softmax(dim=1)(log_probs)
                y_pred = y_prob.argmax(1)
                accuracy = (labels == y_pred).type(torch.float).mean()

                loss = self._val_err_func(log_probs, labels)
                batch_valid_loss += loss.item()
                batch_accuracy += accuracy.item()

            batch_valid_loss /= batch_idx+1
            batch_accuracy /= batch_idx+1

        return batch_valid_loss, batch_accuracy

    def get_conf_mat(self, dataset, model=None):
        if model is None:
            model = self._model
        data_ldr = DataLoader(
            dataset, batch_size=1, shuffle=False)
        device = self.args.device
        nb_classes = len(torch.unique(dataset.targets))
        confusion_matrix = ConfusionMatrix(
            nb_classes, normalize='all'
        ).to(device)
        model.eval()
        with torch.no_grad():

            for inputs, classes in data_ldr:
                inputs = inputs.to(device)
                classes = classes.to(device)
                log_probs = model(inputs)
                preds = nn.Softmax(dim=1)(log_probs).argmax(1)
                confusion_matrix.update(preds, classes)
        return confusion_matrix

    def get_dst_list(self):
        return self._graph.neighbors(self._id)

    def get_max_trust_in_neigh(self):
        trust_list = [self._graph[self._id][n]['weight']
                      for n in list(self._graph.neighbors(self._id))]
        return np.max(trust_list)

    def get_trust_in_neighs(self):
        # everything static so far, so we can compute it at init and use it directly
        trust_dict = {n: self._graph[self._id][n]['weight']
                      for n in list(self._graph.neighbors(self._id))}
        return trust_dict

    def reset_model():
        pass

    def _social_aggregation(self, active_neighs, models, data_sizes):
        
        trust_dict = self.get_trust_in_neighs()
        trust_active_dict = {k: trust_dict[k]
                             for k in active_neighs if k in trust_dict}
        tot_data_neighbourhood = sum(
            [data_sizes[k] for k in active_neighs if k in data_sizes])+self.train_size
        tot_models = len(models)+1

        # prepare models and trust values for averaging
        w_list = [self._model.state_dict()] # the local model
        t_list = [self._selfconfidence]     # the self-confidence (1 default)
        

        if self._args.use_weighted_avg:
            alpha_list = [self.train_size /tot_data_neighbourhood]
        else:
            alpha_list = [1./tot_models]

        for k in models.keys():
            w_list.append(models[k])
            t_list.append(trust_active_dict[k])
            if self._args.use_weighted_avg:
                alpha_list.append(data_sizes[k]/tot_data_neighbourhood)
            else:
                alpha_list.append(1./tot_models)




        # do the averaging
        model_avg = SocialFedAvg(w_list, t_list, alpha_list)
        self._model.load_state_dict(model_avg)

    def _social_diff_aggregation(self, active_neighs, models, data_sizes):

        tot_data_neighbourhood = sum([data_sizes[k] for k in active_neighs if k in data_sizes])
        tot_models = len(models)
        if self._args.include_myself:
            tot_data_neighbourhood+self.train_size
            tot_models+=1

        trust_dict = self.get_trust_in_neighs()
        trust_active_dict = {k: trust_dict[k]
                             for k in active_neighs if k in trust_dict}

        # prepare models, alphas and trust values for averaging
        w_list = []
        t_list = []
        alpha_list = []
        
        for k in models.keys():
            w_list.append(models[k])
            t_list.append(trust_active_dict[k])
            if self._args.use_weighted_avg:
                alpha_list.append(data_sizes[k]/tot_data_neighbourhood)                
            else:
                alpha_list.append(1/tot_models)
        # include myself in the list for computing the average model
        if self._args.include_myself:
            w_list.append(self.model.state_dict())
            t_list.append(self.selfconfidence)
            if self._args.use_weighted_avg:
                alpha_list.append(self.train_size/tot_data_neighbourhood)                
            else:
                alpha_list.append(1./tot_models)
        
        # print('w_list',w_list)
        # print('t_list',t_list)
        # print('alpha_list',alpha_list)
            
            # do the averaging
        model_update = SocialLocalGlobalDiffUpdate(
            local_model=self._model.state_dict(), models=w_list, trust=t_list,alphas=alpha_list)
        self._model.load_state_dict(model_update)
        # return model_update
    




    def _consensus_fa(self,active_neighs, models,data_sizes):

        if self.args.cfa_epsilon == -1:
            cfa_epsilon = 1/len(active_neighs)
        else:
            cfa_epsilon = self.args.cfa_epsilon
        
        tot_data_neightbourhood = sum([data_sizes[k] for k in active_neighs if k in data_sizes])
        alpha_list = []
        w_list = []
        for k in models.keys():
            alpha_list.append(data_sizes[k]/tot_data_neightbourhood)
            w_list.append(models[k])
        
        model_update = consensus_federated_average(local_model=self._model.state_dict(),models=w_list,alphas=alpha_list,epsilon=cfa_epsilon)
        self._model.load_state_dict(model_update)
        # return model_update

    def _fedavg_aggregation(self, models):        
        w_list = []        
        for k in models.keys():
            w_list.append(models[k])            
        model_avg = FedAvg(w_list)
        self._model.load_state_dict(model_avg)
        # return model_avg
    

    def _substitute_local_with_global_model(self, models: dict):
        _, mdl = models.popitem()
        self._model.load_state_dict(mdl)

class savazziPaiv(simplePaiv):
    def __init__(self, id, args, graph, dataset, data_idxs, model, train_role):
        super().__init__(id, args, graph, dataset, data_idxs, model, train_role)
    
        self.psi_model = copy.deepcopy(self._model)
        self._model_previous_round = copy.deepcopy(self._model)
        self.rho = .99
        self.mubeta = 0.0015
        self.neigh_gradients = {}
        # self.neight_prev_models = []
        self._local_strategy = CFAGE_LocalModelUpdate(self._args, dataset=self._dataset, idxs=(
            self.train_idx, self.val_idx), loss_func=self._err_func, val_loss_func=self._val_err_func)


    def train(self, current_time):
        # here we implement the atomic training that includes interaction between local and peers' models
        # workflow:
        # 1) save local model in psi_model
        self.psi_model.load_state_dict(self._model.state_dict())
        # 2) receive models from the other peers
        if self.args.verbose:
            print("PAIV", self._id, "/ The msg_buffer size is:",
                  len(self._msg_buffer.queue))

        # DEBUG: we print the models of selected nodes
        if self._args.toggle_output_model and self._id in [0, 1]:
            filename = f'stats/{self._args.outfolder}/models/model_{self._id}_at_{current_time}_{self._args.seed}.pt'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            print(f'Saving models in {filename}')
            torch.save(self._model, filename)

        active_neighs = list()
        if not self._msg_buffer.empty():
            # implement logic for selecting the models to aggregate
            # simple: use all models in the buffer
            models = {}
            neigh_data_size = {}
            gradients = {}
            # note: the [:] is necessary, it creates a copy of
            for msg in self._msg_buffer.queue[:]:
                # self._msg_buffer.queue and the removal with the .get() is applied
                # only when msg.time < current_time
                if msg.time < current_time:
                    if self.args.verbose:
                        print("TRAIN: adding model from", msg.source,
                              "generated at time", msg.time)
                    # associate model with their source node for weighting
                    models[msg.source] = msg.model.state_dict()
                    neigh_data_size[msg.source] = msg.dataset_size
                    gradients[msg.source] = msg.gradients
                    active_neighs.append(msg.source)
                    self._msg_buffer.get()  # remove the processed message from the message buffer
            if self.args.verbose:
                print("-- ", self._msg_buffer.qsize(),
                      "messages already available for the next training epoch")

        # print(f'Node {self._id} is aggregating the models coming from {active_neighs}')
        

        # save the current model
        self._model_previous_round.load_state_dict(self._model.state_dict())
        # perform merging if you have received messages from your neighbors
        if len(active_neighs)>0:
                # compute aggregate model
                self._consensus_fa(
                        active_neighs=active_neighs, models=models, data_sizes=neigh_data_size)                
                # update gradients 
                self._compute_gradients(models=models)                
                # update local model with psi and gradients
                w1 = self._update_model_with_gradients(active_neighs=active_neighs,gradients=gradients)
                # for k,v in w1.items():
                #     torch._assert(torch.all(torch.logical_not(torch.isnan(v))),f'{k} contains NaNs')
                self._model.load_state_dict(w1)
                
        # DEBUG: we print the models of selected nodes
        if self._args.toggle_output_model and self._id in [0, 1]:
            filename = f'stats/{self._args.outfolder}/models/model_after_feddiff_{self._id}_at_{current_time}_{self._args.seed}.pt'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            print(f'Saving models in {filename}')
            torch.save(self._model, filename)

        # train local model
        w, loss, vloss = self._local_strategy.train(
            net=self._model)
        
        # update the local model if he new validation loss improves over the previous one
        # Note: this works only if the validation set is reliable. Not sure what happens in highly non-iid data distribution.
        always_update = True
        if always_update or vloss < self._best_valid_loss:
            self._best_valid_loss = vloss
            self._model.load_state_dict(w)
        else:
            # revert to the model before the aggregation + retrain
            print("No improvement in valid loss, reverting to backup copy of the model")
            self.model.load_state_dict(self._model_previous_round.state_dict())

        # DEBUG: we print the models of selected nodes
        if self._args.toggle_output_model and self._id in [0, 1]:
            filename = f'stats/{self._args.outfolder}/models/model_after_retrain_{self._id}_at_{current_time}_{self._args.seed}.pt'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            print(f'Saving models in {filename}')
            torch.save(self._model, filename)
        
        # return the loss after the training epoch
        return loss 

    def _consensus_fa(self, active_neighs, models, data_sizes):

        if self.args.cfa_epsilon == -1:
            cfa_epsilon = 1/len(active_neighs)
        else:
            cfa_epsilon = self.args.cfa_epsilon

        tot_data_neightbourhood = sum(
            [data_sizes[k] for k in active_neighs if k in data_sizes])
        alpha_list = []
        w_list = []
        for k in models.keys():
            alpha_list.append(data_sizes[k]/tot_data_neightbourhood)
            w_list.append(models[k])

        model_update = consensus_federated_average(local_model=self.psi_model.state_dict(
        ), models=w_list, alphas=alpha_list, epsilon=cfa_epsilon)
        self.psi_model.load_state_dict(model_update)
    
    def _compute_gradients(self, models):
        tmp_model = copy.deepcopy(self.psi_model)
                
        for k in models.keys():
            tmp_model.load_state_dict(models[k])
            # init the grads to 0, the first time a new neigh is encountered. 
            if k not in self.neigh_gradients.keys():
                self.neigh_gradients[k] = {k:torch.zeros_like(ten) for k,ten in self._model.named_parameters()}
            
            # compute the gradients for the given model. 
            tmp_grad_dict = self._local_strategy.computed_gradient(tmp_model)
            
            # update models' gradients estimate
            for l in models[k].keys():    
                self.neigh_gradients[k][l] = self.rho*tmp_grad_dict[l] +(1-self.rho)*self.neigh_gradients[k][l]

    def _update_model_with_gradients(self, active_neighs:list, gradients:list):
        psi_weights = self.psi_model.state_dict()
        for i in active_neighs:
            if gradients[i] != None:
                for k in psi_weights.keys():
                    psi_weights[k] += -self.mubeta*gradients[i][k]
                    
        return psi_weights

class hessianPaiv(AbstractPaiv):

    def __init__(self, id, args, graph, dataset, data_idxs, model,hess_diag_, train_role):
        super().__init__(id=id, args=args, graph=graph, dataset=dataset, data_idxs=data_idxs,
                         model=model)
        self._msg_buffer = PriorityQueue()

        # training related properties
        self._loss_local = []
        # self._valid_loss = np.Inf

        self._err_func = nn.CrossEntropyLoss(reduction='none')
        self._val_err_func = nn.CrossEntropyLoss()

        # train_size = int(np.floor(len(data_idxs) * (1-self.args.val_split)))
        # randidx = torch.randperm(len(data_idxs))
        # self.train_idx = data_idxs[randidx[:train_size]]
        # self.val_idx = data_idxs[randidx[train_size:]]

        self.train_idx = data_idxs['train']
        self.val_idx = data_idxs['validation']
        self.train_size = len(self.train_idx)
        print(
            f'Data distrib: {torch.unique(dataset.targets[self.val_idx],return_counts=True)}')

        ####################################################################################
        # Hessian diagonal initialization
        #self.neigh_hessian_diag = {}

        self.hessian_diag = hess_diag_
        self.hessian_daig_round_counter = 0

        

        ####################################################################################

        self._local_strategy = StandardLocalUpdateV1_hessian(
            self._args, dataset=self._dataset, idxs=(self.train_idx, self.val_idx), loss_func=self._err_func,val_loss_func=self._val_err_func)

        self._best_valid_loss = np.Inf

        self._train_role = train_role

        self.aggregation_func = self._args.aggregation_func

        self.toggle_aggregation = self.args.toggle_aggregate_first



    def train(self, current_time):
        # here we implement the atomic training that includes interaction between local and peers' models
        # workflow:
        # 1) process peers' models
        # 2) train local model embedding peers' knowledge
        if self.args.verbose:
            print("PAIV", self._id, "/ The msg_buffer size is:",
                  len(self._msg_buffer.queue))
        
        # DEBUG: we print the models of selected nodes
        if self._args.toggle_output_model and self._id in [0,1]:
            filename = f'stats/{self._args.outfolder}/models/model_{self._id}_at_{current_time}_{self._args.seed}.pt'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            print(f'Saving models in {filename}')
            torch.save(self._model, filename)

        active_neighs = list()
        if not self._msg_buffer.empty():
            # implement logic for selecting the models to aggregate
            # simple: use all models in the buffer
            models = {}
            neigh_data_size = {}
            hessian_daigs = {}
            # note: the [:] is necessary, it creates a copy of
            for msg in self._msg_buffer.queue[:]:
                # self._msg_buffer.queue and the removal with the .get() is applied
                # only when msg.time < current_time
                if msg.time < current_time:
                    if self.args.verbose:
                        print("TRAIN: adding model from", msg.source,
                              "generated at time", msg.time)
                    # associate model with their source node for weighting
                    models[msg.source] = msg.model.state_dict()
                    neigh_data_size[msg.source] = msg.dataset_size
                    hessian_daigs[msg.source] = msg.hessian_diag
                    active_neighs.append(msg.source)
                    self._msg_buffer.get()  # remove the processed message from the message buffer
            if self.args.verbose:
                print("-- ", self._msg_buffer.qsize(),
                      "messages already available for the next training epoch")
        
        # print(f'Node {self._id} is aggregating the models coming from {active_neighs}')

        # save the current model
        prev_model = copy.deepcopy(self._model)
        # perform merging if you have received messages from your neighbors
        if active_neighs:
            #self._compute_hessian_diag(models)
            if self._train_role == 'standalone':
                if self.aggregation_func == 'fed_avg':
                    # aggregate models using social federated average aggregation
                    self._social_aggregation(
                        active_neighs=active_neighs, models=models, data_sizes=neigh_data_size)
                elif self.aggregation_func == 'fed_diff_hessian_diag':
                    self._social_diff_aggregation_hessian_diag(
                        active_neighs=active_neighs, models=models, data_sizes=neigh_data_size, hessian_diags=hessian_daigs)

                elif self.aggregation_func == 'cfa':
                    self._consensus_fa(active_neighs=active_neighs, models=models,data_sizes=neigh_data_size)
            elif self._train_role == 'fed_client':
                # simply substitute local with global model
                self._substitute_local_with_global_model(models=models)
        
        # DEBUG: we print the models of selected nodes
        if self._args.toggle_output_model and self._id in [0,1]:
            filename = f'stats/{self._args.outfolder}/models/model_after_feddiff_{self._id}_at_{current_time}_{self._args.seed}.pt'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            print(f'Saving models in {filename}')
            torch.save(self._model, filename)

        # train local model
        w, loss, vloss = self._local_strategy.train(
            net=self._model.to(self._args.device))
        
        # update the local model if the new validation loss improves over the previous one
        if self.hessian_daig_round_counter <= self._args.stop_hess_comm:
            #print('Computing hessian diag')
            hessian_temp_diag = self._local_strategy.computed_hessian_diag(copy.deepcopy(self._model))
            self.hessian_daig_round_counter +=1
        else:
            #print('use existing hessian diagonal info')
            hessian_temp_diag = 0
            
            

        if len(self.hessian_diag) > 0:
            for k in self._model.state_dict().keys():
                if "running_mean" in k or "running_var" in k or "num_batches_tracked" in k:
                    continue
                if self.hessian_daig_round_counter <= self._args.stop_hess_comm: 
                    if self._args.moving_avg:
                        self.hessian_diag[k] =  ((1-self._args.hessian_beta)* self.hessian_diag[k]) + (self._args.hessian_beta * hessian_temp_diag[k])
                    else:
                        self.hessian_diag[k] = self.hessian_diag[k] + (self._args.hessian_beta * hessian_temp_diag[k])
                else:
                    self.hessian_diag[k] = self.hessian_diag[k]
            
        else:    
            if self._args.moving_avg:
                print('Hessian is updated by exp moving average!')
            self.hessian_diag = self._local_strategy.computed_hessian_diag(copy.deepcopy(self._model))

        # update the local model if the new validation loss improves over the previous one
        # Note: this works only if the validation set is reliable. Not sure what happens in highly non-iid data distribution. 
        always_update = True
        if always_update or vloss < self._best_valid_loss:
            self._best_valid_loss = vloss
            self._model.load_state_dict(w)
        else:
            # revert to the model before the aggregation + retrain
            print("No improvement in valid loss, reverting to backup copy of the model")
            self._model.load_state_dict(prev_model.state_dict())

        # DEBUG: we print the models of selected nodes
        if self._args.toggle_output_model and self._id in [0,1]:
            filename = f'stats/{self._args.outfolder}/models/model_after_retrain_{self._id}_at_{current_time}_{self._args.seed}.pt'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            print(f'Saving models in {filename}')
            torch.save(self._model, filename)

        
        # return the loss after the training epoch
        return loss 


    def _compute_hessian_diag(self, models):
        tmp_model = copy.deepcopy(self._model)
                
        for k in models.keys():
            tmp_model.load_state_dict(models[k])

            # compute the gradients for the given model. 
            tmp_hessian_diag_dict = self._local_strategy.computed_hessian_diag(tmp_model)


            # init the grads to 0, the first time a new neigh is encountered. 
            if k not in self.neigh_hessian_diag.keys():
                self.neigh_hessian_diag[k] = None

                # self.neigh_hessian_diag[k] = {k:torch.zeros_like(ten) for k,ten in self._model.named_parameters()}
            
            if self.neigh_hessian_diag[k] == None:
                self.neigh_hessian_diag[k] = tmp_hessian_diag_dict
            else:
                for l in models[k].keys():   
                    if "running_mean" in l or "running_var" in l or "num_batches_tracked" in l:
                        continue
                    self.neigh_hessian_diag[k][l] = tmp_hessian_diag_dict[l] + self.neigh_hessian_diag[k][l]
            
            # # update models' gradients estimate
            # for l in models[k].keys():    
            #     #self.neigh_gradients[k][l] = self.rho*tmp_grad_dict[l] +(1-self.rho)*self.neigh_gradients[k][l]
            #     self.neigh_hessian_diag[k][l] = tmp_hessian_diag_dict[l] #+(1-self.rho)*self.neigh_hessian_diags[k][l]

    def validate(self):
        return self._best_valid_loss

    def test(self, dataset, model=None):
        if model is None:
            model = self._model

        # TODO: here we apply the local model to the validation set
        batch_valid_loss = 0
        batch_accuracy = 0
        ldr_dataset = DataLoader(
            dataset, batch_size=self.args.local_bs)
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(ldr_dataset):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)

                log_probs = model(images)

                y_prob = nn.Softmax(dim=1)(log_probs)
                y_pred = y_prob.argmax(1)
                accuracy = (labels == y_pred).type(torch.float).mean()

                loss = self._val_err_func(log_probs, labels)
                batch_valid_loss += loss.item()
                batch_accuracy += accuracy.item()

            batch_valid_loss /= batch_idx+1
            batch_accuracy /= batch_idx+1

        return batch_valid_loss, batch_accuracy

    def get_conf_mat(self, dataset, model=None):
        if model is None:
            model = self._model
        data_ldr = DataLoader(
            dataset, batch_size=1, shuffle=False)
        device = self.args.device
        nb_classes = len(torch.unique(dataset.targets))
        confusion_matrix = ConfusionMatrix(
            nb_classes, normalize='all'
        ).to(device)
        model.eval()
        with torch.no_grad():

            for inputs, classes in data_ldr:
                inputs = inputs.to(device)
                classes = classes.to(device)
                log_probs = model(inputs)
                preds = nn.Softmax(dim=1)(log_probs).argmax(1)
                confusion_matrix.update(preds, classes)
        return confusion_matrix

    def get_dst_list(self):
        return self._graph.neighbors(self._id)

    def get_max_trust_in_neigh(self):
        trust_list = [self._graph[self._id][n]['weight']
                      for n in list(self._graph.neighbors(self._id))]
        return np.max(trust_list)

    def get_trust_in_neighs(self):
        # everything static so far, so we can compute it at init and use it directly
        trust_dict = {n: self._graph[self._id][n]['weight']
                      for n in list(self._graph.neighbors(self._id))}
        return trust_dict

    def reset_model():
        pass

    def _social_aggregation(self, active_neighs, models, data_sizes):
        
        trust_dict = self.get_trust_in_neighs()
        trust_active_dict = {k: trust_dict[k]
                             for k in active_neighs if k in trust_dict}
        tot_data_neighbourhood = sum(
            [data_sizes[k] for k in active_neighs if k in data_sizes])+self.train_size
        tot_models = len(models)+1

        # prepare models and trust values for averaging
        w_list = [self._model.state_dict()] # the local model
        t_list = [self._selfconfidence]     # the self-confidence (1 default)
        

        if self._args.use_weighted_avg:
            alpha_list = [self.train_size /tot_data_neighbourhood]
        else:
            alpha_list = [1./tot_models]

        for k in models.keys():
            w_list.append(models[k])
            t_list.append(trust_active_dict[k])
            if self._args.use_weighted_avg:
                alpha_list.append(data_sizes[k]/tot_data_neighbourhood)
            else:
                alpha_list.append(1./tot_models)




        # do the averaging
        model_avg = SocialFedAvg(w_list, t_list, alpha_list)
        self._model.load_state_dict(model_avg)

    def _social_diff_aggregation(self, active_neighs, models, data_sizes):

        tot_data_neighbourhood = sum([data_sizes[k] for k in active_neighs if k in data_sizes])
        tot_models = len(models)
        if self._args.include_myself:
            tot_data_neighbourhood+self.train_size
            tot_models+=1

        trust_dict = self.get_trust_in_neighs()
        trust_active_dict = {k: trust_dict[k]
                             for k in active_neighs if k in trust_dict}

        # prepare models, alphas and trust values for averaging
        w_list = []
        t_list = []
        alpha_list = []
        
        for k in models.keys():
            w_list.append(models[k])
            t_list.append(trust_active_dict[k])
            if self._args.use_weighted_avg:
                alpha_list.append(data_sizes[k]/tot_data_neighbourhood)                
            else:
                alpha_list.append(1/tot_models)
        # include myself in the list for computing the average model
        if self._args.include_myself:
            w_list.append(self.model.state_dict())
            t_list.append(self.selfconfidence)
            if self._args.use_weighted_avg:
                alpha_list.append(self.train_size/tot_data_neighbourhood)                
            else:
                alpha_list.append(1./tot_models)
        
        # print('w_list',w_list)
        # print('t_list',t_list)
        # print('alpha_list',alpha_list)
            
            # do the averaging
        model_update = SocialLocalGlobalDiffUpdate_hessian_diag(
            hessian_terms = self.hessian_diag,local_model=self._model.state_dict(), models=w_list, trust=t_list,alphas=alpha_list)
        self._model.load_state_dict(model_update)
        # return model_update
    
    def _social_diff_aggregation_hessian_diag(self, active_neighs, models, data_sizes, hessian_diags):
        #print('Hessian diag aggregation')

        tot_data_neighbourhood = sum([data_sizes[k] for k in active_neighs if k in data_sizes])
        tot_models = len(models)
        if self._args.include_myself:
            tot_data_neighbourhood+self.train_size
            tot_models+=1

        trust_dict = self.get_trust_in_neighs()
        trust_active_dict = {k: trust_dict[k]
                             for k in active_neighs if k in trust_dict}

        # prepare models, alphas and trust values for averaging
        w_list = []
        t_list = []
        alpha_list = []
        hessian_diag_list = []
        
        for k in models.keys():
            w_list.append(models[k])
            t_list.append(trust_active_dict[k])
            # hessian_diag_list.append(self.neigh_hessian_diag[k])
            hessian_diag_list.append(hessian_diags[k])
            if self._args.use_weighted_avg:
                alpha_list.append(data_sizes[k]/tot_data_neighbourhood)                
            else:
                alpha_list.append(1/tot_models)
        # include myself in the list for computing the average model
        if self._args.include_myself:
            w_list.append(self.model.state_dict())
            t_list.append(self.selfconfidence)
            if self._args.use_weighted_avg:
                alpha_list.append(self.train_size/tot_data_neighbourhood)                
            else:
                alpha_list.append(1./tot_models)
        
        summed_hessian_diag = self.sum_hessian_diags(hessian_diag_list)
        normalized_hessian_diags = self.normalize_hessian_diags(hessian_diag_list, summed_hessian_diag, alpha_list)


            
            # do the averaging
        model_update = SocialLocalGlobalDiffUpdate_hessian_diag(
            hessian_terms = normalized_hessian_diags,local_model=self._model.state_dict(), models=w_list, trust=t_list,alphas=alpha_list)
        self._model.load_state_dict(model_update)
        # return model_update



    def _consensus_fa(self,active_neighs, models,data_sizes):

        if self.args.cfa_epsilon == -1:
            cfa_epsilon = 1/len(active_neighs)
        else:
            cfa_epsilon = self.args.cfa_epsilon
        
        tot_data_neightbourhood = sum([data_sizes[k] for k in active_neighs if k in data_sizes])
        alpha_list = []
        w_list = []
        for k in models.keys():
            alpha_list.append(data_sizes[k]/tot_data_neightbourhood)
            w_list.append(models[k])
        
        model_update = consensus_federated_average(local_model=self._model.state_dict(),models=w_list,alphas=alpha_list,epsilon=cfa_epsilon)
        self._model.load_state_dict(model_update)
        # return model_update

    def _fedavg_aggregation(self, models):        
        w_list = []        
        for k in models.keys():
            w_list.append(models[k])            
        model_avg = FedAvg(w_list)
        self._model.load_state_dict(model_avg)
        # return model_avg
    

    def _substitute_local_with_global_model(self, models: dict):
        _, mdl = models.popitem()
        self._model.load_state_dict(mdl)

    def sum_hessian_diags(self,hessian_diags):
        '''
        Sum Hessian diagonal values across a list of model Hessian diagonals.
        Args:
            hessian_diags (list of dict): List where each element is a dictionary of Hessian diagonal values for a model.
        Returns:
            summed_hessian_diag (dict): A dictionary with summed Hessian diagonal values for each parameter.
        '''
        summed_hessian_diag = {}
        
        for hessian_diag in hessian_diags:
            for name, hessian in hessian_diag.items():
                if name not in summed_hessian_diag:
                    summed_hessian_diag[name] = torch.zeros_like(hessian)
                summed_hessian_diag[name] += hessian
        
        return summed_hessian_diag

    # def normalize_hessian_diags(self, hessian_diags, summed_hessian_diag,alpha_list):
    #     '''
    #     Normalize Hessian diagonal values by dividing each Hessian diagonal by the summed Hessian diagonal values.
    #     Args:
    #         hessian_diags (list of dict): List where each element is a dictionary of Hessian diagonal values for a model.
    #         summed_hessian_diag (dict): Dictionary with summed Hessian diagonal values for each parameter.
    #     Returns:
    #         normalized_hessian_diags (list of dict): List of dictionaries with normalized Hessian diagonal values.
    #     '''
    #     normalized_hessian_diags = []

    #     # check if any element is 0  replace with 1
    #     for name, hessian in summed_hessian_diag.items():
    #         summed_hessian_diag[name] = torch.where(hessian == 0, torch.ones_like(hessian), hessian)


    #     for hessian_diag in hessian_diags:
    #         normalized_hessian_diag = {}
    #         for name, hessian in hessian_diag.items():
                
    #             normalized_hessian_diag[name] = hessian / summed_hessian_diag[name]
    #         normalized_hessian_diags.append(normalized_hessian_diag)
        
    #     return normalized_hessian_diags



    def normalize_hessian_diags(self, hessian_diags, summed_hessian_diag,alpha_list):
        '''
        Normalize Hessian diagonal values by dividing each Hessian diagonal by the summed Hessian diagonal values.
        Args:
            hessian_diags (list of dict): List where each element is a dictionary of Hessian diagonal values for a model.
            summed_hessian_diag (dict): Dictionary with summed Hessian diagonal values for each parameter.
        Returns:
            normalized_hessian_diags (list of dict): List of dictionaries with normalized Hessian diagonal values.
        '''
        normalized_hessian_diags = []

        # # check if any element is 0  replace with 1
        # for name, hessian in summed_hessian_diag.items():
        #     summed_hessian_diag[name] = torch.where(hessian == 0, torch.ones_like(hessian), hessian)


        for i, hessian_diag in enumerate(hessian_diags):
            normalized_hessian_diag = {}
            for name, hessian in hessian_diag.items():

                # Create a mask for where summed Hessian is zero
                mask = summed_hessian_diag[name] == 0
                normalized_hessian_diag[name] = torch.where(mask, 0 * torch.ones_like(hessian), hessian / summed_hessian_diag[name])
                # normalized_hessian_diag[name] = torch.where(mask, alpha_list[i] * torch.ones_like(hessian), hessian / summed_hessian_diag[name])
                
                #normalized_hessian_diag[name] = hessian / summed_hessian_diag[name]
            normalized_hessian_diags.append(normalized_hessian_diag)
        
        return normalized_hessian_diags


