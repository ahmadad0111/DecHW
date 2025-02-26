#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import random
import copy
import numpy as np
import torch
from sklearn import metrics
from torch import autograd,nn
from torch.utils.data import DataLoader, Dataset, Sampler, Subset,BatchSampler,SubsetRandomSampler
from torch.utils.data.dataset import random_split
import torch.nn.functional as F
from torchmetrics import ConfusionMatrix


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

    

        
class StandardLocalUpdate(object):
    '''
    The class implements the standard local training functions.  
    '''

    def __init__(self, args, dataset=None, idxs=None, loss_func=None, val_loss_func=None, optimizer_type="SGD"):
        self.args = args
        
        self.loss_func = loss_func
        
        self.val_loss_func = val_loss_func
        if self.val_loss_func is None:
            self.val_loss_func = loss_func
        
        self.selected_clients = []
        self.optimizer_type = optimizer_type
        # local dataset split of train and validation
        
        torch._assert(len(
            idxs) == 2, '[ERROR] StandardLocalUpdate:idxs should be a tuple of lenght 2: (train_idxs,valid_idxs)')
        self.dataset = dataset
        self.train_idx = idxs[0]
        self.valid_idx = idxs[1]
        
        # load train and validation set
        self.ldr_train = DataLoader(
            Subset(dataset,self.train_idx), batch_size=self.args.local_bs, shuffle=True)
        self.ldr_valid = DataLoader(
            Subset(dataset, self.valid_idx), batch_size=self.args.local_bs)
        
        self.unique_labels = torch.unique(self.dataset.targets[self.train_idx])
        self.nb_classes = len(self.unique_labels)
        #self.nb_classes = self.args.num_classes

    def train(self, net):
        net.train()
        # save local copy of net for rollback when increasing validation loss
        prev_net_state = net.state_dict()
        # train and update !! implements a stateless strategy
        if self.optimizer_type == "SGD":
            optimizer = torch.optim.SGD(
                net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        elif self.optimizer_type == "Adam":
            optimizer = torch.optim.Adam(net.parameters(),lr=self.args.lr,weight_decay=self.args.weight_decay)
        

        epoch_loss = 0

        current_valid_loss = np.Inf
        for iter in range(self.args.local_ep):
            batch_loss = 0
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images =  images.to(self.args.device)
                labels = labels.to(self.args.device)
                
                optimizer.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx *
                        len(images), len(self.ldr_train.dataset),
                        100. * batch_idx / len(self.ldr_train), loss.item()))
            
                batch_loss += loss.item()
            epoch_loss += batch_loss/(batch_idx+1)

            new_valid_loss, new_acc = self.validation(net, self.ldr_valid)
            if self.args.verbose:
                print(
                    'Validation Loss: {:.6f}\tAccuracy: {:.6f}'.format(new_valid_loss, new_acc))

            if new_valid_loss > current_valid_loss or new_valid_loss == 0:
                if self.args.verbose:
                    print('Early stop at local epoch {}: for increasing valid loss [new:{:.6f} > prev:{:.6f}]'.format(
                        iter, new_valid_loss, current_valid_loss))
                # save last valid loss (the increased one)

                # rollback to prev best net
                # net = copy.deepcopy(prev_net)
                net.load_state_dict(prev_net_state)
                current_valid_loss = new_valid_loss
                break
            # else:

                # backup current net before next epoch
                # prev_net = copy.deepcopy(net)
            current_valid_loss = new_valid_loss
        epoch_loss /= iter+1

        
        return net.state_dict(), epoch_loss, current_valid_loss

    def validation(self, net, dataset,conf_mat=False):
        batch_valid_loss = 0
        batch_accuracy = 0

        
        

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataset):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)

                log_probs = net(images)
                y_prob = nn.Softmax(dim=1)(log_probs)
                y_pred = y_prob.argmax(1)
                accuracy = (labels == y_pred).type(torch.float).mean()

                # loss = self.loss_func(log_probs, labels)
                loss = self.val_loss_func(log_probs, labels)
                batch_valid_loss += loss.item()
                batch_accuracy += accuracy.item()
            batch_valid_loss /= batch_idx+1
            batch_accuracy /= batch_idx+1
        
        return batch_valid_loss, batch_accuracy

    def get_confusion_matrix(self, net, dataset):
        device = self.args.device
        
        confusion_matrix = ConfusionMatrix(
            self.nb_classes, normalize='all'
        ).to(device)
        net.eval()
        with torch.no_grad():

            for inputs, classes in dataset:
                inputs = inputs.to(device)
                classes = classes.to(device)
                log_probs = net(inputs)
                preds = nn.Softmax(dim=1)(log_probs).argmax(1)
                confusion_matrix.update(preds, classes)

        return confusion_matrix.compute()





class StandardLocalUpdateV1(object):
    '''
    The class implements the standard local training functions conditionally regularized
    '''

    def __init__(self, args, dataset=None, idxs=None, loss_func=None, val_loss_func=None):
        self.args = args

        self.loss_func = loss_func

        self.kl_loss = nn.KLDivLoss(reduction='none')

        self.val_loss_func = val_loss_func

        
        if self.val_loss_func is None:
            self.val_loss_func = loss_func

        self.selected_clients = []
        self.optimizer_type = self.args.optimizer
        # local dataset split of train and validation
        
        torch._assert(len(
            idxs) == 2, '[ERROR] StandardLocalUpdate:idxs should be a tuple of lenght 2: (train_idxs,valid_idxs)')
        self.dataset = dataset
        self.train_idx = idxs[0]
        self.valid_idx = idxs[1]

        # load train and validation set
        self.ldr_train = DataLoader(
            Subset(dataset, self.train_idx), batch_size=self.args.local_bs, shuffle=True)
        self.ldr_valid = DataLoader(
            Subset(dataset, self.valid_idx), batch_size=self.args.local_bs)

        self.unique_labels = torch.unique(self.dataset.targets[self.train_idx])
        self.nb_classes = len(self.args.num_classes)
        #self.nb_classes = self.args.num_classes
        

        

    def train(self, net):
        net.train()
        
        # save local copy of net for rollback when increasing validation loss
        # TODO: this can be removed because now we save the prev net before the aggregation
        prev_net = copy.deepcopy(net)
        # train and update !! implements a stateless strategy
        if self.optimizer_type == "SGD":
            optimizer = torch.optim.SGD(
                net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        elif self.optimizer_type == "Adam":
            optimizer = torch.optim.Adam(
                net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        current_valid_loss = np.Inf

        for iter in range(self.args.local_ep):
            if self.args.verbose:
                print(f'Current epoch: {iter}')
            batch_loss = 0
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                # if self.args.verbose:
                #     print(f'- Current batch:{batch_idx}')
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)
                
                output = net(images)
                torch._assert(torch.logical_not(torch.any(torch.isnan(output))),'output in train() is NaN')
                # Loss with ERM +  KL Term
                if self.args.kd_alpha > 0:
                    # s_prob = torch.log_softmax(output,dim=1)
                    # Self KD 
                    one_hot = F.one_hot(labels, num_classes=self.nb_classes)
                    if self.args.vteacher_generator == 'fixed':
                        t_prob = self.args.skd_beta * one_hot + ((1-one_hot)*(1-self.args.skd_beta)/(self.nb_classes-1))
                    if self.args.vteacher_generator == 'random':
                        batch_size = labels.shape[0]
                        sample = torch.distributions.Uniform(
                            self.args.skd_beta, 1.).sample((batch_size, 1))
                        rsample = sample.repeat((1, self.nb_classes)).to(self.args.device)
                        
                        t_prob =  one_hot*rsample + (1-one_hot)*(1-rsample)/(self.nb_classes-1)

                    
                    kl = self.kl_loss(torch.log_softmax(output,dim=1),t_prob)

                    loss = torch.nanmean((1-self.args.kd_alpha) * self.loss_func(output, labels) + self.args.kd_alpha * torch.sum(kl,dim=1),dim=0)
                    # if self.args.verbose:
                    #     print(f'Batch combined loss: {loss}')
                else:
                    # Loss with ERM only
                    loss = torch.nanmean(self.loss_func(output, labels), dim=0)
                    # torch._assert(torch.logical_not(torch.isnan(loss)),'Loss in train() is NaN')
                    # if self.args.verbose:
                    #     print(f'Batch ERM loss: {loss}')
                
                # print(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # if self.args.verbose and batch_idx % 10 == 0:
                #     print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         iter, batch_idx *
                #         len(images), len(self.ldr_train.dataset),
                #         100. * batch_idx / len(self.ldr_train), loss.item()))
                # torch._assert(torch.logical_not(torch.isnan(loss)),'Loss in train() is NaN')
                batch_loss += loss.item()
            if self.args.verbose:
                print(f'All batches of current epoch DONE')
            epoch_loss = batch_loss/(batch_idx+1)
            if self.args.verbose:
                print(f'Epoch loss: {epoch_loss}')

            new_valid_loss, new_acc = self.validation(net, self.ldr_valid)
            if self.args.verbose:
                print(
                    'Validation Loss: {:.6f}\tAccuracy: {:.6f}'.format(new_valid_loss, new_acc))

            if new_valid_loss > current_valid_loss or new_valid_loss == 0:
                if self.args.verbose:
                    print('Early stop at local epoch {}: for increasing valid loss [new:{:.6f} > prev:{:.6f}]'.format(
                        iter, new_valid_loss, current_valid_loss))
                # save last valid loss (the increased one)

                # rollback to prev best net
                net = copy.deepcopy(prev_net)
                # current_valid_loss = new_valid_loss
                break
            else:

                # backup current net before next epoch
                # prev_net = copy.deepcopy(net)
                current_valid_loss = new_valid_loss
        epoch_loss /= self.args.local_ep
        return net.state_dict(), epoch_loss, current_valid_loss

    # # TODO make this function indedependent
    # def validation(self, net, dataset):
    #     batch_valid_loss = []
    #     with torch.no_grad():
    #         net.eval()
    #         for batch_idx, (images, labels) in enumerate(dataset):
    #             images, labels = images.to(
    #                 self.args.device), labels.to(self.args.device)

    #             log_probs = net(images)
    #             loss = self.loss_func(log_probs, labels)
    #             batch_valid_loss.append(loss.item())
    #     return sum(batch_valid_loss)/len(batch_valid_loss)

    def validation(self, net, dataset, conf_mat=False):
        batch_valid_loss = 0
        batch_accuracy = 0
        net.eval()
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataset):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)

                log_probs = net(images)
                y_prob = nn.Softmax(dim=1)(log_probs)
                y_pred = y_prob.argmax(1)
                accuracy = (labels == y_pred).type(torch.float).mean()

                # loss = self.loss_func(log_probs, labels)
                loss = self.val_loss_func(log_probs, labels)
                batch_valid_loss += loss.item()
                batch_accuracy += accuracy.item()
            batch_valid_loss /= batch_idx+1
            batch_accuracy /= batch_idx+1

        return batch_valid_loss, batch_accuracy

    def get_confusion_matrix(self, net, dataset):
        device = self.args.device

        confusion_matrix = ConfusionMatrix(
            self.nb_classes, normalize='all'
        ).to(device)
        net.eval()
        with torch.no_grad():

            for inputs, classes in dataset:
                inputs = inputs.to(device)
                classes = classes.to(device)
                log_probs = net(inputs)
                preds = nn.Softmax(dim=1)(log_probs).argmax(1)
                confusion_matrix.update(preds, classes)

        return confusion_matrix.compute()

class CFAGE_LocalModelUpdate(StandardLocalUpdateV1):
    '''
    PAIV implementing the logic for local model update of Algorithm 2 of the paper: 
    "SAVAZZI et al.: FEDERATED LEARNING WITH COOPERATING DEVICES: CONSENSUS APPROACH FOR MASSIVE IoT NETWORKS"
    '''
    def __init__(self, args, dataset=None, idxs=None, loss_func=None, val_loss_func=None):
        super().__init__(args, dataset, idxs, loss_func, val_loss_func)

        
    
    def computed_gradient(self,net)->dict:
        '''
            Implmentation of line 9, Algorithm 2 of the paper. It returns a dictionary key:grad 
            
        '''
        net.eval()
        net.zero_grad()
        for images, labels in self.ldr_train:
            images, labels = images.to(
                self.args.device), labels.to(self.args.device)

            log_probs = net(images)
            # y_prob = nn.Softmax(dim=1)(log_probs)
            # y_pred = y_prob.argmax(1)
            

            # loss = self.loss_func(log_probs, labels)
            loss = self.val_loss_func(log_probs, labels)
            loss.backward()
            break
        # current_grad = [copy.deepcopy(v.grad) for v in net.parameters()]
        current_grad = {k:v.grad for k,v in net.named_parameters()}
        for v in current_grad.values():
            torch._assert(torch.all(torch.logical_not(torch.isnan(v))), 'NaN in gradients') 
        
        return current_grad


    
        
            



# TODO: remove, not used
class DistillationLocalUpdate(object):
    '''
    The class implements the standard local training functions.  
    '''

    def __init__(self, args, dataset=None, idxs=None, loss_func=None, val_loss_func = None, optimizer_type="SGD"):
        self.args = args
        self.loss_func = loss_func
        self.val_loss_func = val_loss_func
        if self.val_loss_func is None:
            self.val_loss_func = loss_func
        self.selected_clients = []
        self.optimizer_type = optimizer_type
        # local dataset split of train and validation
        self.dataset = dataset
        torch._assert(len(
            idxs) == 2, '[ERROR] StandardLocalUpdate:idxs should be a tuple of lenght 2: (train_idxs,valid_idxs)')
        self.train_idx = idxs[0]
        self.valid_idx = idxs[1]

        self.local_valid = Subset(dataset, self.valid_idx)
        self.local_train = Subset(dataset, self.train_idx)

        self.ldr_valid = DataLoader(
            self.local_valid, batch_size=self.args.local_bs)
        self.ldr_train = DataLoader(
            self.local_train, batch_size=self.args.local_bs, shuffle=True) # this had bs=1
        
        self.local_idxs = set(self.train_idx)
        # load validation set
        self.kl_loss = nn.KLDivLoss(reduction='batchmean',log_target=True)
       

        if self.args.verbose:
            print(f'Dataset{torch.unique(self.dataset.targets[self.train_idx], return_counts=True)}')
            print(f'Dataset{torch.unique(self.dataset.targets[self.valid_idx], return_counts=True)}')
            
            
            
        # exit(0)
        

        self.unique_labels = torch.unique(self.dataset.targets[self.train_idx]).tolist()
        self.nb_classes = len(self.unique_labels)
        self.local_performance = None
        self.lookup_teachers = None

        self.train_confusion_matrix = ConfusionMatrix(
            self.nb_classes, normalize='true'
        ).to(self.args.device)

    # TODO: define the train function according to the teacher specialties. 
    def train(self, net, teachers,alpha=0.5):
        net.train()
        # save local copy of net for rollingback when increasing validation loss
        prev_net = copy.deepcopy(net)
        
        # update the teachers 
        for teacher in teachers:
            self.update_teacher(teacher)
        epoch_loss = 0
        # for each teacher
        current_valid_loss = np.Inf

        
        if self.optimizer_type == "SGD":
            optimizer = torch.optim.SGD(
                net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
              

        loss = 0               
        for iter in range(self.args.local_distil_ep):
            epoch_loss = 0       
            n_batch = 0    
            kl = 0
            kl_count = 0
            erm = 0
            erm_count = 0
            
            for sample_idx, (patt, lbl) in enumerate(self.ldr_train):
                images, labels = patt.to(
                    self.args.device), lbl.to(self.args.device)                                
                
                output = net(images)
                          

                if self.lookup_teachers[labels.item()] is not None:
                    teacher = self.lookup_teachers[labels.item()]                    
                    teacher.eval()
                    with torch.no_grad():
                        teacher_probs = teacher(images)
                
                # DISTILLATION LOSS
                
                #   'Teacher KL term not a probability'
                    s_prob = torch.log_softmax(output, dim=1)
                    t_prob = torch.log_softmax(teacher_probs, dim=1)
                    kl += self.kl_loss(s_prob,t_prob)
                    kl_count +=1                   
                else:                   
                    erm += self.loss_func(output, labels)
                    erm_count +=1
                   
                

                if sample_idx != 0 and sample_idx % self.args.local_bs == 0:
                    optimizer.zero_grad()
                    loss = (1-alpha)*erm/(erm_count+1e-24) + alpha* kl/(kl_count+1e-24)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss
                    n_batch +=1
                    kl = 0
                    erm = 0
                    erm_count = 0
                    kl_count = 0
            # assert(n_batch>0,"Actual batch size = 0 ")
            epoch_loss/=n_batch
            if self.args.verbose:
                print('Update Epoch: {} \tLoss: {:.6f}'.format(
                    iter, epoch_loss.item()))
        # batch_loss += loss.item()
        # epoch_loss += batch_loss/(batch_idx+1)
            


        # if len(self.lookup_teachers.keys()) > 0:
        #     epoch_loss /= len(self.lookup_teachers.keys())
        
            new_valid_loss, new_acc,cmat = self.validate(net, self.ldr_valid,conf_mat=True)

            if self.args.verbose:
                print(
                    'Validation Loss: {:.6f}\tAccuracy: {:.6f}\nConf mat.: {}'.format(new_valid_loss, new_acc,cmat.diag()))
                

            if new_valid_loss > current_valid_loss or new_valid_loss == 0:
                if self.args.verbose:
                    print('Early stop at local epoch {}: for increasing valid loss [new:{:.6f} > prev:{:.6f}]'.format(
                        iter, new_valid_loss, current_valid_loss))
                # save last valid loss (the increased one)

                # rollback to prev best net
                net = copy.deepcopy(prev_net)
                
                
            else:

                # backup current net before next epoch
                current_valid_loss = new_valid_loss
                prev_net = copy.deepcopy(net)
            
        return net.state_dict(), epoch_loss.item(), current_valid_loss

    
    def get_confusion_matrix(self,net,dataset):
        device = self.args.device
        
        self.train_confusion_matrix.reset()

        net.eval()
        
        with torch.no_grad():
            
            for inputs, classes in dataset:
                inputs = inputs.to(device)
                classes = classes.to(device)
                log_probs = net(inputs)
                preds = nn.Softmax(dim=1)(log_probs).argmax(1)
                self.train_confusion_matrix.update(preds,classes)

            
        return self.train_confusion_matrix.compute()
    
    def init_teachers(self,net):
        # set the current 
        self.local_performance = self.get_confusion_matrix(net,self.ldr_valid).diag()
        self.lookup_teachers = {l: None for l in self.unique_labels}
    
    def update_teacher(self, net):
    
        # test the performance of the potential teacher and fill the lookup table 
        local_perf  = self.get_confusion_matrix(net,self.ldr_valid).diag()
        
        
        for lbl in self.unique_labels:
            if local_perf[lbl] > self.local_performance[lbl]:
                self.local_performance[lbl] = local_perf[lbl]
                self.lookup_teachers[lbl] = net
    
    # def update_teachers(self, nets):

        
        
    #     local_perf = self.get_confusion_matrix(nets[0], self.ldr_valid).diag()
    #     if len(nets) > 1:
    #         for idx in range(1,len(nets)):
    #             local_perf = torch.vstack(
    #                 (local_perf, self.get_confusion_matrix(nets[idx], self.ldr_valid).diag()))
    #     max_perf,candidates = torch.max(local_perf,dim=0)
    #     for lbl in self.unique_labels:
    #         if max_perf[lbl] > self.local_performance[lbl]:
    #             self.local_performance[lbl] = max_perf[lbl]
    #             self.lookup_teachers[lbl] = nets[candidates[lbl]]


    

    def validate(self, net, dataset,conf_mat=False):
        
        batch_valid_loss = 0
        batch_accuracy = 0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataset):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)

                log_probs = net(images)
                y_prob = nn.Softmax(dim=1)(log_probs)
                y_pred = y_prob.argmax(1)
                accuracy = (labels == y_pred).type(torch.float).mean()

                loss = self.val_loss_func(log_probs, labels)
                batch_valid_loss += loss.item()
                batch_accuracy += accuracy.item()
            batch_valid_loss /= batch_idx+1
            batch_accuracy /= batch_idx+1
        if conf_mat:
            return batch_valid_loss, batch_accuracy, self.get_confusion_matrix(net,dataset)

        return batch_valid_loss, batch_accuracy
    

class SelfDistillationLocalUpdate(object):
    '''
    The class implements the standard local training functions.  
    '''

    def __init__(self, args, dataset=None, idxs=None, loss_func=None, val_loss_func=None):
        self.args = args
        self.loss_func = loss_func
        self.val_loss_func = val_loss_func
        if self.val_loss_func is None:
            self.val_loss_func = loss_func
        self.selected_clients = []
        self.optimizer_type = self.args.optimizer
        # local dataset split of train and validation
        self.dataset = dataset
        torch._assert(len(
            idxs) == 2, '[ERROR] StandardLocalUpdate:idxs should be a tuple of lenght 2: (train_idxs,valid_idxs)')
        self.train_idx = idxs[0]
        self.valid_idx = idxs[1]

        self.local_train = Subset(dataset, self.train_idx)
        self.local_valid = Subset(dataset, self.valid_idx)
        
        self.ldr_train = DataLoader(self.local_train, batch_size=self.args.local_bs, shuffle=True)
        self.ldr_valid = DataLoader(self.local_valid, batch_size=self.args.local_bs)

        # self.local_idxs = set(self.train_idx)

        # load validation set
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

        if self.args.verbose:
            print(
                f'Dataset{torch.unique(self.dataset.targets[self.train_idx], return_counts=True)}')
            print(
                f'Dataset{torch.unique(self.dataset.targets[self.valid_idx], return_counts=True)}')

        # exit(0)

        self.unique_labels = torch.unique(
            self.dataset.targets[self.train_idx]).tolist()
        self.nb_classes = len(self.unique_labels)
        # self.local_performance = None
        # self.lookup_teachers = None

    # TODO: define the train function according to the teacher specialties.
    def train(self, net):
        net.train()
        # save local copy of net for rollingback when increasing validation loss
        prev_net = copy.deepcopy(net)

        # update the teachers
        # for teacher in teachers:
        #     self.update_teacher(teacher)
        epoch_loss = 0
        # for each teacher
        current_valid_loss = np.Inf

        if self.optimizer_type == "SGD":
            optimizer = torch.optim.SGD(
                net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        elif self.optimizer_type == "Adam":
            optimizer = torch.optim.Adam(
                net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        

        loss = 0
        
        for iter in range(self.args.local_distil_ep):
            if self.args.verbose:
                print(f'Current epoch: {iter}')

            epoch_loss = 0
            n_batch = 0

            for patt, lbl in self.ldr_train:
                if self.args.verbose:
                    print(f'- Current batch:{n_batch}')
                images, labels = patt.to(
                    self.args.device), lbl.to(self.args.device)

                output = net(images)
                # OLD Lorenzo interpretation
                # teacher_probs = (beta*F.one_hot(lbl, num_classes=self.nb_classes) + (1-beta)*torch.rand(1)).to(self.args.device)

                # DISTILLATION LOSS

                #   'Teacher KL term not a probability'
                s_prob = torch.log_softmax(output, dim=1)
                
                # ++ NEW CVPR paper self virtual teacher
                # t_prob = torch.log((beta*F.one_hot(lbl, num_classes=self.nb_classes) +
                #           (1-beta)/(self.nb_classes-1))).to(self.args.device)
                t_prob = self._generate_teacher_probs(
                    labels, beta=self.args.skd_beta, method=self.args.vteacher_generator).to(self.args.device)
                
                # ++ OLD Lorenzo "interpretation"
                # t_prob = torch.log_softmax(teacher_probs, dim=1)

                # ++ ALTERNATE PROBABILISTIC from Lorenzo
                # t_prob = torch.log((beta*F.one_hot(lbl, num_classes=self.nb_classes) +
                #                     (1-beta)/(self.nb_classes-1))).to(self.args.device)
                t_prob_entropy = 0
                if self.args.vteacher_generator == 'random':
                    t_prob_entropy = torch.mean(
                        torch.distributions.Categorical(t_prob).entropy())
                erm = self.loss_func(output, labels)
                # print(f'Batch ERM loss: {erm}')
                kl = self.kl_loss(s_prob, t_prob) + t_prob_entropy
                loss = (1-self.args.kd_alpha)* erm + self.args.kd_alpha*kl
                if self.args.verbose:
                    print(f'Batch combined loss: {loss}')
                # print(loss)
                               
                optimizer.zero_grad()                
                loss.backward()
                optimizer.step()
                epoch_loss += loss
                n_batch += 1

            if self.args.verbose:
                print(f'All batches of current epoch DONE')
                
            # # assert(n_batch>0,"Actual batch size = 0 ")
            epoch_loss /= n_batch
            if self.args.verbose:
                print(f'Epoch loss: {epoch_loss}')
            if self.args.verbose:
                print('Update Epoch: {} \tLoss: {:.6f}'.format(
                    iter, epoch_loss.item()))
        

            new_valid_loss, new_acc = self.validate(
                net, self.ldr_valid, conf_mat=False)

            if self.args.verbose:            
                # print(
                #     'Validation Loss: {:.6f}\tAccuracy: {:.6f}\nConf mat.: {}'.format(new_valid_loss, new_acc, cmat.diag()))
                print(
                    'Validation Loss: {:.6f}\tAccuracy: {:.6f}'.format(new_valid_loss, new_acc))

            if new_valid_loss > current_valid_loss or new_valid_loss == 0:
                if self.args.verbose:
                    print('Early stop at local epoch {}: for increasing valid loss [new:{:.6f} > prev:{:.6f}]'.format(
                        iter, new_valid_loss, current_valid_loss))
                # save last valid loss (the increased one)

                # rollback to prev best net
                net = copy.deepcopy(prev_net)
                break

            else:

                # backup current net before next epoch
                current_valid_loss = new_valid_loss
                prev_net = copy.deepcopy(net)

        return net.state_dict(), epoch_loss.item(), current_valid_loss
    
    def _generate_teacher_probs(self,lbl,beta,method='fixed'):
        
        
        if method == 'fixed':
            # As in CVPR paper
            one_hot = F.one_hot(lbl, num_classes=self.nb_classes)
            return (beta * one_hot +
             (1-one_hot)*(1-beta)/(self.nb_classes-1))
        
        if method == 'random':
            batch_size = lbl.shape[0]
            sample = torch.distributions.Uniform(beta, .99).sample((batch_size,1))
            rsample = sample.repeat((1,self.nb_classes)).to(self.args.device)
            one_hot = F.one_hot(lbl, num_classes=self.nb_classes).to(self.args.device)
            return one_hot*rsample + (1-one_hot)*(1-rsample)/(self.nb_classes-1)



    
    def get_confusion_matrix(self, net, dataset):
        device = self.args.device

        confusion_matrix = ConfusionMatrix(
            self.nb_classes, normalize='true'
        ).to(device)

        net.eval()

        with torch.no_grad():

            for inputs, classes in dataset:
                inputs = inputs.to(device)
                classes = classes.to(device)
                log_probs = net(inputs)
                preds = nn.Softmax(dim=1)(log_probs).argmax(1)
                confusion_matrix.update(preds, classes)

        return confusion_matrix.compute()


   

    def validate(self, net, dataset, conf_mat=False):

        batch_valid_loss = 0
        batch_accuracy = 0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataset):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)

                log_probs = net(images)
                y_prob = nn.Softmax(dim=1)(log_probs)
                y_pred = y_prob.argmax(1)
                accuracy = (labels == y_pred).type(torch.float).mean()

                loss = self.val_loss_func(log_probs, labels)
                batch_valid_loss += loss.item()
                batch_accuracy += accuracy.item()
            batch_valid_loss /= batch_idx+1
            batch_accuracy /= batch_idx+1
        if conf_mat:
            return batch_valid_loss, batch_accuracy, self.get_confusion_matrix(net, dataset)

        return batch_valid_loss, batch_accuracy

# This is used in the oracle-based strategy
class IdealDistillationLocalUpdate(object):
    '''
    The class implements the standard local training functions.  
    '''

    def __init__(self, args, dataset=None, idxs=None, loss_func=None, val_loss_func=None):
        self.args = args
        self.loss_func = loss_func
        self.val_loss_func = val_loss_func
        if self.val_loss_func is None:
            self.val_loss_func = loss_func
        self.selected_clients = []
        self.optimizer_type = self.args.optimizer
        # local dataset split of train and validation
        self.dataset = dataset
        torch._assert(len(
            idxs) == 2, '[ERROR] StandardLocalUpdate:idxs should be a tuple of lenght 2: (train_idxs,valid_idxs)')
        self.train_idx = idxs[0]
        self.valid_idx = idxs[1]

        self.local_valid = Subset(dataset, self.valid_idx)
        self.local_train = Subset(dataset, self.train_idx)

        self.ldr_valid = DataLoader(
            self.local_valid, batch_size=self.args.local_bs)
        self.ldr_train = DataLoader(
            self.local_train, batch_size=self.args.local_bs, shuffle=True)

        self.local_idxs = set(self.train_idx)
        # load validation set
        self.kl_loss = nn.KLDivLoss(reduction='batchmean', log_target=True)

        if self.args.verbose:
            print(
                f'Dataset{torch.unique(self.dataset.targets[self.train_idx], return_counts=True)}')
            print(
                f'Dataset{torch.unique(self.dataset.targets[self.valid_idx], return_counts=True)}')

        # exit(0)

        self.unique_labels = torch.unique(
            self.dataset.targets[self.train_idx]).tolist()
        self.nb_classes = len(self.unique_labels)
        # self.local_performance = None
        # self.lookup_teachers = None

    # TODO: define the train function according to the teacher specialties.
    def train(self, net, teacher,alpha=0.3):
        net.train()
        # save local copy of net for rollingback when increasing validation loss
        prev_net = copy.deepcopy(net)

        epoch_loss = 0

        current_valid_loss = np.Inf

        if self.optimizer_type == "SGD":
            optimizer = torch.optim.SGD(
                net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        elif self.optimizer_type == "Adam":
            optimizer = torch.optim.Adam(
                net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        loss = 0
        teacher.eval()
        for iter in range(self.args.local_distil_ep):
            epoch_loss = 0
            n_batch = 0
        
            
            for patt, lbl in self.ldr_train:
                images, labels = patt.to(
                    self.args.device), lbl.to(self.args.device)

                output = net(images)
                with torch.no_grad():
                    teacher_probs = teacher(images)
                # DISTILLATION LOSS

                #   'Teacher KL term not a probability'
                s_prob = torch.log_softmax(output, dim=1)
                t_prob = torch.log_softmax(teacher_probs, dim=1)

                loss = (1-alpha)*self.loss_func(output, labels) + alpha*self.kl_loss(s_prob, t_prob)
                               
                optimizer.zero_grad()                
                loss.backward()
                optimizer.step()
                epoch_loss += loss
                n_batch += 1
                
            # assert(n_batch>0,"Actual batch size = 0 ")
            epoch_loss /= n_batch
            if self.args.verbose:
                print('Update Epoch: {} \tLoss: {:.6f}'.format(
                    iter, epoch_loss.item()))
        

        new_valid_loss, new_acc, cmat = self.validate(
            net, self.ldr_valid, conf_mat=True)

        if self.args.verbose:
            print(
                'Validation Loss: {:.6f}\tAccuracy: {:.6f}\nConf mat.: {}'.format(new_valid_loss, new_acc, cmat.diag()))

        if new_valid_loss > current_valid_loss or new_valid_loss == 0:
            if self.args.verbose:
                print('Early stop at local epoch {}: for increasing valid loss [new:{:.6f} > prev:{:.6f}]'.format(
                    iter, new_valid_loss, current_valid_loss))
            # save last valid loss (the increased one)

            # rollback to prev best net
            net = copy.deepcopy(prev_net)

        else:

            # backup current net before next epoch
            current_valid_loss = new_valid_loss

        return net.state_dict(), epoch_loss.item(), current_valid_loss

    
    def get_confusion_matrix(self, net, dataset):
        device = self.args.device

        confusion_matrix = ConfusionMatrix(
            self.nb_classes, normalize='true'
        ).to(device)

        net.eval()

        with torch.no_grad():

            for inputs, classes in dataset:
                inputs = inputs.to(device)
                classes = classes.to(device)
                log_probs = net(inputs)
                preds = nn.Softmax(dim=1)(log_probs).argmax(1)
                confusion_matrix.update(preds, classes)

        return confusion_matrix.compute()

   

    def validate(self, net, dataset, conf_mat=False):

        batch_valid_loss = 0
        batch_accuracy = 0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataset):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)

                log_probs = net(images)
                y_prob = nn.Softmax(dim=1)(log_probs)
                y_pred = y_prob.argmax(1)
                accuracy = (labels == y_pred).type(torch.float).mean()

                loss = self.val_loss_func(log_probs, labels)
                batch_valid_loss += loss.item()
                batch_accuracy += accuracy.item()
            batch_valid_loss /= batch_idx+1
            batch_accuracy /= batch_idx+1
        if conf_mat:
            return batch_valid_loss, batch_accuracy, self.get_confusion_matrix(net, dataset)

        return batch_valid_loss, batch_accuracy

# DEPRECATED. used by VirtualDistillationPaiv, which is not used anymore AFAIK
class VirtualDistillationLocalUpdate(object):
    '''
    The class implements the virtual distillation using aggregate
    '''
    # TODO: to be completed 
    def __init__(self, args, dataset=None, idxs=None, loss_func=None, val_loss_func=None, optimizer_type="SGD"):
        self.args = args
        self.loss_func = loss_func
        self.val_loss_func = val_loss_func
        if self.val_loss_func is None:
            self.val_loss_func = loss_func
        self.selected_clients = []
        self.optimizer_type = optimizer_type
        # local dataset split of train and validation
        self.dataset = dataset
        torch._assert(len(
            idxs) == 2, '[ERROR] StandardLocalUpdate:idxs should be a tuple of lenght 2: (train_idxs,valid_idxs)')
        self.train_idx = idxs[0]
        self.valid_idx = idxs[1]

        self.local_valid = Subset(dataset, self.valid_idx)
        self.local_train = Subset(dataset, self.train_idx)
        self.ldr_valid = DataLoader(
            self.local_valid, batch_size=self.args.local_bs)
        self.ldr_train = DataLoader(
            self.local_train, batch_size=self.args.local_bs, shuffle=True)

        self.local_idxs = set(self.train_idx)
        # load validation set
        self.kl_loss = nn.KLDivLoss(reduction='batchmean', log_target=True)

        if self.args.verbose:
            print(
                f'Dataset{torch.unique(self.dataset.targets[self.train_idx], return_counts=True)}')
            print(
                f'Dataset{torch.unique(self.dataset.targets[self.valid_idx], return_counts=True)}')

        # exit(0)

        self.unique_labels = torch.unique(
            self.dataset.targets[self.train_idx]).tolist()
        self.nb_classes = len(self.unique_labels)
        self.local_performance = None
        self.teachers = None

        self.soft_label=0.9

    # TODO: define the train function according to the teacher specialties.
    def train(self, net, teachers_map, alpha=0.5):
        net.train()
        patience = 3
        # save local copy of net for rollingback when increasing validation loss
        if patience == 3:
            prev_net = copy.deepcopy(net)

        # update the teachers
        

        epoch_loss = 0
        # for each teacher
        current_valid_loss = np.Inf

        if self.optimizer_type == "SGD":
            optimizer = torch.optim.SGD(
                net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        loss = 0
        for iter in range(self.args.local_distil_ep):
            epoch_loss = 0
            n_batch = 0
            kl = 0
            

            for patt, lbl in self.ldr_train:
                images, labels = patt.to(
                    self.args.device), lbl.to(self.args.device)

                output = net(images)                
                
                t_probs = torch.vstack(
                    [teachers_map[l] for l in lbl.tolist()])
                s_prob = torch.log_softmax(output, dim=1)

                kl = self.kl_loss(s_prob, t_probs)
                erm = self.loss_func(output, labels)
                
                loss = (1-alpha)*erm + alpha*kl
                
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss
                n_batch += 1
                    
            # assert(n_batch>0,"Actual batch size = 0 ")
            epoch_loss /= n_batch
            if self.args.verbose:
                print('Update Epoch: {} \tLoss: {:.6f}'.format(
                    iter, epoch_loss.item()))


        if self.args.verbose:
            new_valid_loss, new_acc, cmat = self.validate(
                net, self.ldr_valid, conf_mat=True)
        else:
            new_valid_loss, new_acc = self.validate(
                net, self.ldr_valid)

        if self.args.verbose:
            print(
                'Validation Loss: {:.6f}\tAccuracy: {:.6f}\nConf mat.: {}'.format(new_valid_loss, new_acc, cmat.diag()))

        if (new_valid_loss > current_valid_loss or new_valid_loss == 0):
            patience -=1

            if self.args.verbose and patience == 0 :
                print('Early stop at local epoch {}: for increasing valid loss [new:{:.6f} > prev:{:.6f}]'.format(
                    iter, new_valid_loss, current_valid_loss))
            # save last valid loss (the increased one)

            # rollback to prev best net
            if patience ==0:
                net = copy.deepcopy(prev_net)

        else:

            # backup current net before next epoch
            current_valid_loss = new_valid_loss

        return net.state_dict(), epoch_loss.item(), current_valid_loss

    def get_confusion_matrix(self, net, dataset):
        device = self.args.device

        confusion_matrix = ConfusionMatrix(
            self.nb_classes, normalize='true'
        ).to(device)

        net.eval()

        with torch.no_grad():

            for inputs, classes in dataset:
                inputs = inputs.to(device)
                classes = classes.to(device)
                log_probs = net(inputs)
                preds = nn.Softmax(dim=1)(log_probs).argmax(1)
                confusion_matrix.update(preds, classes)

        return confusion_matrix.compute()

   

    def validate(self, net, dataset, conf_mat=False):

        batch_valid_loss = 0
        batch_accuracy = 0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataset):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)

                log_probs = net(images)
                y_prob = nn.Softmax(dim=1)(log_probs)
                y_pred = y_prob.argmax(1)
                accuracy = (labels == y_pred).type(torch.float).mean()

                loss = self.val_loss_func(log_probs, labels)
                batch_valid_loss += loss.item()
                batch_accuracy += accuracy.item()
            batch_valid_loss /= batch_idx+1
            batch_accuracy /= batch_idx+1
        if conf_mat:
            return batch_valid_loss, batch_accuracy, self.get_confusion_matrix(net, dataset)

        return batch_valid_loss, batch_accuracy
    
    def get_local_aggregate_soft_labels(self,model,init=False):
        # return the local sum of output and the correspoding number of occurencies, on a per-label fashion
        aggregate_probs = {l: torch.zeros((self.nb_classes,)).to(
            self.args.device) for l in self.unique_labels}
        aggr_den = {l: torch.zeros((1,)).to(
            self.args.device) for l in self.unique_labels}
        if not init:
            model.eval()
            
            for (img, lbl) in self.ldr_train:
                images, labels = img.to(self.args.device), lbl.to(self.args.device)
                with torch.no_grad():
                    output = F.softmax(model(images),dim=1)
                for lab in lbl.tolist():
                    label_position = torch.flatten(torch.nonzero(labels == lab)).tolist()
                    
                    aggregate_probs[lab] += torch.sum(output[label_position].squeeze(), dim=0)
                    aggr_den[lab] += len(label_position)
                
            for lab in aggregate_probs.keys():
                    aggregate_probs[lab] = aggregate_probs[lab]/aggr_den[lab]
                    
            

        return aggregate_probs,aggr_den
class StandardLocalUpdateV1_hessian(object):
    '''
    The class implements the standard local training functions conditionally regularized
    '''

    def __init__(self, args, dataset=None, idxs=None, loss_func=None, val_loss_func=None):
        self.args = args

        self.loss_func = loss_func

        self.kl_loss = nn.KLDivLoss(reduction='none')

        self.val_loss_func = val_loss_func

        
        if self.val_loss_func is None:
            self.val_loss_func = loss_func

        self.selected_clients = []
        self.optimizer_type = self.args.optimizer
        # local dataset split of train and validation
        
        torch._assert(len(
            idxs) == 2, '[ERROR] StandardLocalUpdate:idxs should be a tuple of lenght 2: (train_idxs,valid_idxs)')
        self.dataset = dataset
        self.train_idx = idxs[0]
        self.valid_idx = idxs[1]

        # load train and validation set
        self.ldr_train = DataLoader(
            Subset(dataset, self.train_idx), batch_size=self.args.local_bs, shuffle=True)
        self.ldr_valid = DataLoader(
            Subset(dataset, self.valid_idx), batch_size=self.args.local_bs)

        self.unique_labels = torch.unique(self.dataset.targets[self.train_idx])
        self.nb_classes = self.args.num_classes

        

    def train(self, net):
        net.train()
        
        # save local copy of net for rollback when increasing validation loss
        # TODO: this can be removed because now we save the prev net before the aggregation
        prev_net = copy.deepcopy(net)
        # train and update !! implements a stateless strategy
        if self.optimizer_type == "SGD":
            optimizer = torch.optim.SGD(
                net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        elif self.optimizer_type == "Adam":
            optimizer = torch.optim.Adam(
                net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        current_valid_loss = np.Inf


        for iter in range(self.args.local_ep):
            if self.args.verbose:
                print(f'Current epoch: {iter}')
            batch_loss = 0
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                # if self.args.verbose:
                #     print(f'- Current batch:{batch_idx}')
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)
                
                output = net(images)
                torch._assert(torch.logical_not(torch.any(torch.isnan(output))),'output in train() is NaN')
                # Loss with ERM +  KL Term
                if self.args.kd_alpha > 0:
                    # s_prob = torch.log_softmax(output,dim=1)
                    # Self KD 
                    one_hot = F.one_hot(labels, num_classes=self.nb_classes)
                    if self.args.vteacher_generator == 'fixed':
                        t_prob = self.args.skd_beta * one_hot + ((1-one_hot)*(1-self.args.skd_beta)/(self.nb_classes-1))
                    if self.args.vteacher_generator == 'random':
                        batch_size = labels.shape[0]
                        sample = torch.distributions.Uniform(
                            self.args.skd_beta, 1.).sample((batch_size, 1))
                        rsample = sample.repeat((1, self.nb_classes)).to(self.args.device)
                        
                        t_prob =  one_hot*rsample + (1-one_hot)*(1-rsample)/(self.nb_classes-1)

                    
                    kl = self.kl_loss(torch.log_softmax(output,dim=1),t_prob)

                    loss = torch.nanmean((1-self.args.kd_alpha) * self.loss_func(output, labels) + self.args.kd_alpha * torch.sum(kl,dim=1),dim=0)
                    # if self.args.verbose:
                    #     print(f'Batch combined loss: {loss}')
                else:
                    # Loss with ERM only
                    loss = torch.nanmean(self.loss_func(output, labels), dim=0)
                    # torch._assert(torch.logical_not(torch.isnan(loss)),'Loss in train() is NaN')
                    # if self.args.verbose:
                    #     print(f'Batch ERM loss: {loss}')
                
                # print(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # # get the hessian diagonal
                # for name, param in net.named_parameters():
                #     if param.requires_grad:
                #         self.hessian_diag[name] += param.grad.pow(2).detach()/len(labels)

                # if self.args.verbose and batch_idx % 10 == 0:
                #     print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         iter, batch_idx *
                #         len(images), len(self.ldr_train.dataset),
                #         100. * batch_idx / len(self.ldr_train), loss.item()))
                # torch._assert(torch.logical_not(torch.isnan(loss)),'Loss in train() is NaN')
                batch_loss += loss.item()
            if self.args.verbose:
                print(f'All batches of current epoch DONE')
            epoch_loss = batch_loss/(batch_idx+1)
            if self.args.verbose:
                print(f'Epoch loss: {epoch_loss}')

            new_valid_loss, new_acc = self.validation(net, self.ldr_valid)
            if self.args.verbose:
                print(
                    'Validation Loss: {:.6f}\tAccuracy: {:.6f}'.format(new_valid_loss, new_acc))

            if new_valid_loss > current_valid_loss or new_valid_loss == 0:
                if self.args.verbose:
                    print('Early stop at local epoch {}: for increasing valid loss [new:{:.6f} > prev:{:.6f}]'.format(
                        iter, new_valid_loss, current_valid_loss))
                # save last valid loss (the increased one)

                # rollback to prev best net
                net = copy.deepcopy(prev_net)
                # current_valid_loss = new_valid_loss
                break
            else:

                # backup current net before next epoch
                # prev_net = copy.deepcopy(net)
                current_valid_loss = new_valid_loss
        epoch_loss /= self.args.local_ep
        return net.state_dict(), epoch_loss, current_valid_loss

    def computed_hessian_diag(self,net)->dict:
        '''
            Implmentation of line 9, Algorithm 2 of the paper. It returns a dictionary key:grad 
            
        '''
        net.eval()
        
        # get the hessian diagonal
        hessian_diag = {name: torch.zeros_like(param) for name, param in net.named_parameters() if param.requires_grad}
        #into device
        hessian_diag = {k:v.to(self.args.device) for k,v in hessian_diag.items()}



        for images, labels in self.ldr_train:
            net.zero_grad()
            images, labels = images.to(
                self.args.device), labels.to(self.args.device)

            log_probs = net(images)
            # y_prob = nn.Softmax(dim=1)(log_probs)
            # y_pred = y_prob.argmax(1)
            

            # loss = self.loss_func(log_probs, labels)
            loss = self.val_loss_func(log_probs, labels)
            loss.backward()
            # hessian diag
            for name, param in net.named_parameters():
                if param.requires_grad:
                    hessian_diag[name] += param.grad.pow(2).detach()#/len(labels)
                
                # norm = torch.norm(hessian_diag[name])
                # if norm != 0:
                #     hessian_diag[name] = hessian_diag[name]/norm

                
            hessian_diag = {k:v.to(self.args.device) for k,v in hessian_diag.items()}
        
        return hessian_diag


    # def computed_hessian_diag(self,net)->dict:
    #     '''
    #         Implmentation of line 9, Algorithm 2 of the paper. It returns a dictionary key:grad 
            
    #     '''
    #     net.eval()
    #     net.zero_grad()
    #     # get the hessian diagonal
    #     hessian_diag = {name: torch.zeros_like(param) for name, param in net.named_parameters() if param.requires_grad}
    #     #into device
    #     hessian_diag = {k:v.to(self.args.device) for k,v in hessian_diag.items()}



    #     for images, labels in self.ldr_train:
    #         images, labels = images.to(
    #             self.args.device), labels.to(self.args.device)

    #         log_probs = net(images)
    #         # y_prob = nn.Softmax(dim=1)(log_probs)
    #         # y_pred = y_prob.argmax(1)
            

    #         # loss = self.loss_func(log_probs, labels)
    #         loss = self.val_loss_func(log_probs, labels)
    #         loss.backward()
    #         # hessian diag
    #         for name, param in net.named_parameters():
    #             if param.requires_grad:
    #                 hessian_diag[name] += param.grad.pow(2).detach()#/len(labels)
                
    #             norm = torch.norm(hessian_diag[name])
    #             if norm != 0:
    #                 hessian_diag[name] = hessian_diag[name]/norm

                
    #         hessian_diag = {k:v.to(self.args.device) for k,v in hessian_diag.items()}
        
    #     return hessian_diag



    # # TODO make this function indedependent
    # def validation(self, net, dataset):
    #     batch_valid_loss = []
    #     with torch.no_grad():
    #         net.eval()
    #         for batch_idx, (images, labels) in enumerate(dataset):
    #             images, labels = images.to(
    #                 self.args.device), labels.to(self.args.device)

    #             log_probs = net(images)
    #             loss = self.loss_func(log_probs, labels)
    #             batch_valid_loss.append(loss.item())
    #     return sum(batch_valid_loss)/len(batch_valid_loss)

    def validation(self, net, dataset, conf_mat=False):
        batch_valid_loss = 0
        batch_accuracy = 0
        net.eval()
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataset):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)

                log_probs = net(images)
                y_prob = nn.Softmax(dim=1)(log_probs)
                y_pred = y_prob.argmax(1)
                accuracy = (labels == y_pred).type(torch.float).mean()

                # loss = self.loss_func(log_probs, labels)
                loss = self.val_loss_func(log_probs, labels)
                batch_valid_loss += loss.item()
                batch_accuracy += accuracy.item()
            batch_valid_loss /= batch_idx+1
            batch_accuracy /= batch_idx+1

        return batch_valid_loss, batch_accuracy

    def get_confusion_matrix(self, net, dataset):
        device = self.args.device

        confusion_matrix = ConfusionMatrix(
            self.nb_classes, normalize='all'
        ).to(device)
        net.eval()
        with torch.no_grad():

            for inputs, classes in dataset:
                inputs = inputs.to(device)
                classes = classes.to(device)
                log_probs = net(inputs)
                preds = nn.Softmax(dim=1)(log_probs).argmax(1)
                confusion_matrix.update(preds, classes)

        return confusion_matrix.compute()
