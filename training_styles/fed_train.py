import copy
from training_strategies.aggregation_strategy import FedAvg,WeightedFedAvg
from torch.utils import data
from models import MLP
from paiv import simplePaiv
from clock import Clock
from message import Message
from icecream import ic
import utils.utils as uutils
import csv
import os
import torch

class FederatedLearning:

    def __init__(self, args, g, dataset_train, data_partitions, dataset_test, local_models) -> None:
        self._args = args
        self._graph = g.sai_graph
        self._center_id = -1
        # I assign only clock event for the central server, which then triggers all the PAIVs
        # the central server id is set to -1
        self._clock = Clock(
            [self._center_id], max_comm_rounds=self._args.communication_rounds)

        # this are used to store the currently processed event info during the runs
        self._current_time = None
        self._current_node = None
        self._event_type = None
        self._event_counter = None

        # server loss
        self._train_loss_server = []
        self._validation_loss_server = []

        # setting up the paivs
        self._paiv_list = []
        self._train_loss = {}
        self._validation_loss = {}
        self._test_loss = {}
        self._test_accuracy = {}

        self._dataset_test = dataset_test
        # self._global_model = None
        role = 'fed_client'
        
        self._global_model = local_models[0]

        for pid in list(self._graph.nodes):
            # # set it in training mode
            # self._global_model.train()

            paiv = simplePaiv(
                id=pid, args=self._args, graph=self._graph, dataset=dataset_train, data_idxs=data_partitions[pid], model=copy.deepcopy(self._global_model), train_role=role)
            self._paiv_list.append(paiv)
            self._train_loss[pid] = []
            self._test_loss[pid] = []
            self._validation_loss[pid] = []
            self._test_accuracy[pid] = []
        
        # DEBUG: print the existing PAIVs
        if self._args.verbose:
            print("FED_TRAIN: We have ", len(self._paiv_list), "PAIVs")
            for p in self._paiv_list:
                print("\tI'm paiv", p.id, "and my local data contains",
                      len(p.data_idxs), "items")

    @property
    def paiv_list(self):
        return self._paiv_list

    @property
    def train_loss(self):
        return self._train_loss

    @property
    def validation_loss(self):
        return self._validation_loss

    def run(self):
        while not self._clock.empty():
            (self._current_time, self._current_node, self._event_type,
             self._event_counter) = self._clock.get_next()
            if self._args.verbose:
                print("FED_TRAIN: Time=", self._current_time,
                      " - serving node", self._current_node)

            # train
            self.activate_paiv_training()

            # the resulting local model is sent to selected destination nodes
            self.collect_and_update()

            self.stop_if_condition_met()

        # print final train loss
        self.write_stats()

    def models_are_different(self, mdl1, mdl2):

        return torch.sum(torch.tensor([torch.norm(k1-k2, p=2).item()
                                       for k1, k2 in zip(mdl1.parameters(), mdl2.parameters())]))

    def check_models(self):
        print("Check models are different")
        for i in range(len(self._paiv_list)-1):
            for j in range(i+1, len(self._paiv_list)):
                print(
                    f'Paiv {i}, Paiv {j};{self.models_are_different(self._paiv_list[i]._model, self._paiv_list[j]._model)}')

    def activate_paiv_training(self):
        for p in self._paiv_list:

            paiv_loss_train = p.train(self._current_time)
            paiv_loss_valid = p.validate()
            paiv_loss_test, paiv_loss_acc = p.test(self._dataset_test)

            self._train_loss[p.id].append(paiv_loss_train)
            self._test_loss[p.id].append(paiv_loss_test)
            self._test_accuracy[p.id].append(paiv_loss_acc)
            self._validation_loss[p.id].append(paiv_loss_valid)

    def collect_and_update(self):
        # collect models from paiv_list and use them for training
        # set self._train_loss_server and self._validation_loss_server
        models = []
        alphas = []
        for p in self._paiv_list:
            models.append(p.model.state_dict())
            alphas.append(p.train_size)

        
        if self._args.use_weighted_avg:
            tot_alpha = sum(alphas)
            alphas = [i/tot_alpha for i in alphas]
            avg_model = WeightedFedAvg(models,alphas)
        else:
            avg_model = FedAvg(models)

        self._global_model.load_state_dict(avg_model)
        # manda il nuovo modello ai paiv
        for p in self._paiv_list:
            p.msg_buffer.put(
                Message(self._current_time, self._center_id, self._global_model))

    def stop_if_condition_met(self):
        # check the stopping condition
        if self._args.verbose:
            print("Check the stopping condition")
        if not self._clock.training_finished(self._event_counter + 1):
            # if training not done yet, set up the next clock event for the node
            self._clock.push_next(time=self._current_time + 1, node=self._current_node, event_type="TRAIN",
                                  event_counter=self._event_counter + 1)
        else:
            print("Node", self._current_node,
                  "is done (either reached max_epochs or not improving loss anymore)")

    def write_stats(self):
        filename = "stats/" + self._args.outfolder + \
            "/loss_fed_" + str(self._args.seed) + ".tsv"
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
