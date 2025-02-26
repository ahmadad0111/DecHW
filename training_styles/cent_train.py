from icecream import ic
import numpy as np
from torch.utils.data.dataset import random_split
from torch.utils.data.dataloader import DataLoader
# from torch.utils.data import Dataset, Sampler, Subset,BatchSampler,SubsetRandomSampler
from torch.utils.data import Subset
import torch
import torch.nn.functional as F
# from models import CNNMnist, MLP
from torch import nn
from numpy.lib.function_base import append
import os
#from os import WIFCONTINUED
import copy
import csv
from utils.sampling import get_idxs_per_class
from torchmetrics import ConfusionMatrix

def get_validation_balanced(args, dataset_train, data_idxs):
    labels_with_idx = get_idxs_per_class(dataset_train)
    
    # the size of the training set, based on the input val_split
    train_size = int(np.floor(len(data_idxs) * (1-args.val_split)))


class CentralisedLearning:
    def __init__(self, args, dataset_train, data_partitions, dataset_test, local_models) -> None:
        self._args = args
        self._dataset_train = dataset_train
        self._dataset_test = dataset_test

        # TODO: split the dataset_train into train and validation
        # n_images = self._dataset_train.data.shape[0]

        # # flatten the idxs into a single tensor (= the central server train set is the union of the training set of individual nodes)
        # data_idxs = torch.cat([data_partitions[node] for node, val in data_partitions.items()]) 
        # print(data_idxs.size())
        # # the size of the training set, based on the input val_split
        # train_size = int(np.floor(len(data_idxs) * (1-self._args.val_split)))
        # randidx = torch.randperm(len(data_idxs)) # permute the idxs 
        # self.train_idx = data_idxs[randidx[:train_size]] # get the first train_size as training set
        # self.val_idx = data_idxs[randidx[train_size:]] # put the rest into validation set

        # flatten the idxs into a single tensor (= the central server train set is the union of the training set of individual nodes)
        # self.train_idx = torch.cat([idxs for idxs in data_partitions['train'].values()]) 
        # self.val_idx = torch.cat([idxs for idxs in data_partitions['validation'].values()]) 

        self.train_idx = torch.cat([data_partitions[u]['train'] for u in data_partitions.keys()]) 
        self.val_idx = torch.cat([data_partitions[u]['validation'] for u in data_partitions.keys()]) 

        # print(list(data_partitions['train'].values()))
        # print(self.train_idx.tolist())

        print(f'The train set size is {self.train_idx.size()} and the validation set size is { self.val_idx.size()}')
        print(f'Train data distrib: {torch.unique(dataset_train.targets[self.train_idx],return_counts=True)}')
        print(f'Valid data distrib: {torch.unique(dataset_train.targets[self.val_idx],return_counts=True)}')


        # load train and validation set and test set
        self.ldr_train = DataLoader(
            Subset(dataset_train, self.train_idx), batch_size=self._args.bs, shuffle=True)
        self.ldr_valid = DataLoader(
            Subset(dataset_train, self.val_idx), batch_size=self._args.bs)
        self.ldr_test = DataLoader(
            dataset_test, batch_size=self._args.bs)
         

        self._loss_func = nn.CrossEntropyLoss(reduction='none')
        self.kl_loss = nn.KLDivLoss(reduction='none')

        # len_in = 1
        # for x in dataset_train[0][0].shape:
        #     len_in *= x

        # self._net = MLP(dim_in=len_in, dim_out=self._args.num_classes).to(
        #     self._args.device)
        self._nb_classes = self._args.num_classes
        self._net = local_models[0]

        self._train_loss = {0: []}
        self._validation_loss = {0: []}
        self._test_loss = {0: []}
        self._test_accuracy = {0: []}

    def loss_fn_ce_with_skd(self, output, labels):
        if self._args.kd_alpha > 0:
            
            # Self KD
            one_hot = F.one_hot(labels, num_classes=self._nb_classes)
            if self._args.vteacher_generator == 'fixed':
                t_prob = self._args.skd_beta * one_hot + \
                    ((1-one_hot)*(1-self._args.skd_beta)/(self._nb_classes-1))
            if self._args.vteacher_generator == 'random':
                batch_size = labels.shape[0]
                sample = torch.distributions.Uniform(
                    self._args.skd_beta, 1.).sample((batch_size, 1))
                rsample = sample.repeat(
                    (1, self._nb_classes)).to(self._args.device)

                t_prob = one_hot*rsample + \
                    (1-one_hot)*(1-rsample)/(self._nb_classes-1)

            kl = self.kl_loss(torch.log_softmax(output, dim=1), t_prob)

            loss = torch.mean((1-self._args.kd_alpha) * self._loss_func(
                output, labels) + self._args.kd_alpha * torch.sum(kl, dim=1), dim=0)
            if self._args.verbose:
                print(f'Batch combined loss: {loss}')
        else:
            # Loss with ERM only
            loss = torch.mean(self._loss_func(output, labels), dim=0)
            if self._args.verbose:
                print(f'Batch ERM loss: {loss}')
        return loss

    def run(self):
        # run the training, ignoring decentralisation completely
        self.centralised_train()

        # write the stats
        self.write_stats()

        # save the centralised global model
        self.save_model()

    

    def centralised_train(self):
        self._net.train()

        optimizer = torch.optim.SGD(
            self._net.parameters(), lr=self._args.lr, momentum=self._args.momentum)
        # optimizer = torch.optim.Adam(self._net.parameters(), weight_decay=0.05)

        epoch_loss = []
        valid_loss = []
        test_loss = []
        test_accuracy = []
        current_valid_loss = np.Inf
        

        for iter in range(self._args.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self._args.device), labels.to(self._args.device)
                # print(images.size())
                # print(labels.size())
                optimizer.zero_grad()
                output = self._net(images)
                # ---- Old version
                #  loss = self._loss_func(log_probs, labels)
                # ----

                # --- New ERM + V-KD composite loss function tuned using kd_alpha
                # Loss with ERM +  KL Term
                loss = self.loss_fn_ce_with_skd(output,labels)


                loss.backward()
                optimizer.step()

                if self._args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tValid: {:.6f}'.format(
                        iter, batch_idx *
                        len(images), len(self.ldr_train.dataset),
                        100. * batch_idx / len(self.ldr_train), loss.item(), self.validation(self._net, self.ldr_valid)[1]))
                batch_loss.append(loss.item())
            
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

            # validation

            new_valid_loss, new_acc = self.validation(
                self._net, self.ldr_valid)
            if self._args.verbose:
                print(f'Validation Loss: {new_valid_loss:.6f}')

            print(
                'Epoch {} - Validation Loss: {:.6f}\tAccuracy: {:.6f}'.format(iter, new_valid_loss, new_acc))
            if not self._args.no_early_stop and new_valid_loss > current_valid_loss or new_valid_loss == 0:
                if self._args.verbose:
                    print('Early stop at local epoch {}: for increasing valid loss [new:{:.6f} > prev:{:.6f}]'.format(
                        iter, new_valid_loss, current_valid_loss))
                # save last valid loss (the increased one)

                # rollback to prev best net
                self._net = copy.deepcopy(prev_net)
                # current_valid_loss = new_valid_loss
                valid_loss.append(current_valid_loss)
                break
            else:
                # backup current net before next epoch
                prev_net = copy.deepcopy(self._net)
            
                current_valid_loss = new_valid_loss
                valid_loss.append(current_valid_loss)
            # tloss, tacc = self.validation(self._net, self.ldr_test)
            tloss, tacc = self.test(self._net, self.ldr_test)
            test_loss.append(tloss)
            test_accuracy.append(tacc)
            # print(f'Epoch: {iter} / Validation loss: {current_valid_loss} / Test loss: {tloss} / Test accuracy: {tacc}')
        
        # return epoch_loss, current_valid_loss
        self._train_loss[0] = epoch_loss
        self._validation_loss[0] = valid_loss
        self._test_loss[0] = test_loss
        self._test_accuracy[0] = test_accuracy

    def validation(self, net, dataset):
        batch_valid_loss = 0
        batch_accuracy = 0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataset):
                images, labels = images.to(
                    self._args.device), labels.to(self._args.device)

                log_probs = net(images)
                y_prob = nn.Softmax(dim=1)(log_probs)
                y_pred = y_prob.argmax(1)
                accuracy = (labels == y_pred).type(torch.float).mean()

                loss = self.loss_fn_ce_with_skd(log_probs, labels).item()
                batch_valid_loss += loss
                batch_accuracy += accuracy.item()
            batch_valid_loss /= batch_idx+1
            batch_accuracy /= batch_idx+1

        return batch_valid_loss, batch_accuracy
    
    def test(self, net, dataset):
        batch_valid_loss = 0
        batch_accuracy = 0
        all_targets = torch.IntTensor().to(self._args.device)
        all_preds = torch.IntTensor().to(self._args.device)
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataset):
                images, labels = images.to(
                    self._args.device), labels.to(self._args.device)
                all_targets = torch.cat((all_targets,labels))
                log_probs = net(images)
                y_prob = nn.Softmax(dim=1)(log_probs)
                y_pred = y_prob.argmax(1)
                all_preds = torch.cat((all_preds, y_pred))
                accuracy = (labels == y_pred).type(torch.float).mean()
                loss = self.loss_fn_ce_with_skd(log_probs, labels).item()
                batch_valid_loss += loss
                batch_accuracy += accuracy.item()
            batch_valid_loss /= batch_idx+1
            batch_accuracy /= batch_idx+1

        n_classes = len(self._dataset_test.class_to_idx)
        # n_classes = all_targets.unique().size(dim=0)
        # all_preds = torch.sub(all_preds, 1) # needed because the labels are in [1, x] instead of 
        # all_targets = torch.sub(all_targets, 1)
        # print(all_preds)
        # print(all_targets)
        confmat = ConfusionMatrix(task = 'multiclass', num_classes=n_classes).to(self._args.device)
        cm = confmat(all_preds, all_targets)
        # cm = confmat(all_preds[0:10], all_targets[0:10])
        return batch_valid_loss, batch_accuracy

    def write_stats(self):
        # write the train and validation loss as in the DEC case?

        filename = "stats/" + self._args.outfolder + \
            "/loss_centr_" + str(self._args.seed) + ".tsv"
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'w') as f:
            wr = csv.writer(f)
            wr.writerow(['nodeid', 'time', 'loss', 'loss_type'])
            for k in self._train_loss.keys():
                for t in range(len(self._train_loss[k])):
                    wr.writerow([k, t, self._train_loss[k][t], 'train'])

            for k in self._validation_loss.keys():
                for t in range(len(self._validation_loss[k])):
                    wr.writerow(
                        [k, t, self._validation_loss[k][t], 'validation'])

            for k in self._test_loss.keys():
                for t in range(len(self._test_loss[k])):
                    wr.writerow([k, t, self._test_loss[k][t], 'test'])

            for k in self._test_accuracy.keys():
                for t in range(len(self._test_accuracy[k])):
                    wr.writerow([k, t, self._test_accuracy[k][t], 'accuracy'])
    
    def save_model(self):
        filename = "stats/" + self._args.outfolder + \
            "/centralised_model_" + str(self._args.seed) + ".pt"
        
        torch.save(self._net,filename)

    
