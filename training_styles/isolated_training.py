from models import MLP
from paiv import simplePaiv
from clock import Clock
from message import Message
from icecream import ic
import utils.utils as uutils
import csv
import os
import torch
from os.path import exists

class IsolatedLearning:
    """" Manages the learning process

    This is where the clock and the paivs are located. Also where we keep track of how the learning is going.
    """

    def __init__(self, args, g, dataset_train, data_partitions, dataset_test, local_models) -> None:
        self._args = args
        self._graph = g.sai_graph
        self._clock = Clock(list(self._graph.nodes),
                            max_comm_rounds=self._args.communication_rounds)

        # this are used to store the currently processed event info during the runs
        self._current_time = None
        self._current_node = None
        self._event_type = None
        self._event_counter = None

        self._paiv_list = []
        self._train_loss = {}
        self._validation_loss = {}
        self._test_loss = {}
        self._test_accuracy = {}

        for pid in list(self._graph.nodes):
            # len_in = 1
            # print(type(dataset_train))
            # for x in dataset_train[0][0].shape:
            #     len_in *= x
            # local_model = MLP(
            #     dim_in=len_in, dim_out=args.num_classes).to(args.device)
            # # set it in training mode
            # local_model.train()
            paiv = simplePaiv(
                id=pid, args=self._args, graph=self._graph, dataset=dataset_train, data_idxs=data_partitions[pid], model=local_models[pid], train_role='standalone')
            self._paiv_list.append(paiv)
            self._train_loss[pid] = []
            self._validation_loss[pid] = []
            self._test_loss[pid] = []
            self._test_accuracy[pid] = []
        
        self._dataset_test = dataset_test
        # DEBUG: print the existing PAIVs
        if self._args.verbose:
            print("ISOLATION_TRAIN: We have ", len(self._paiv_list), "PAIVs")
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
        # do until there are not events in the clock
        while not self._clock.empty():
            (self._current_time, self._current_node, self._event_type,
             self._event_counter) = self._clock.get_next()
            if self._args.verbose:
                print("Time:", self._current_time,
                    " - serving node", self._current_node)
            
            if self._args.toggle_model_dist and self._current_time % 5 == 0:
                filename = f'stats/{self._args.outfolder}/models_isolation_at_{self._current_time}.pt'
                if not exists(filename):
                    dist_table = self.check_models()

                    # print(f'Saving models distances in {filename}')
                    torch.save(torch.tensor(dist_table), filename)

            # train
            paiv_loss_train = self._paiv_list[self._current_node].train(
                self._current_time)
            self._train_loss[self._current_node].append(paiv_loss_train)
            # TODO: swap below, used only for testing while waiting for the validation training
            self._validation_loss[self._current_node].append(
                self._paiv_list[self._current_node].validate())

            tloss, tacc = self._paiv_list[self._current_node].test(
                self._dataset_test)
            self._test_loss[self._current_node].append(tloss)
            self._test_accuracy[self._current_node].append(tacc)
            if self._args.verbose:
                print("Done with training")

            ## NOTE: we don't send the models to the neighbors, so the local nodes simply trains its model
            # the resulting local model is sent to selected destination nodes
            #self.send_local_model()

            self.stop_if_condition_met()

        # print final train loss
        ic(self._train_loss)
        self.write_stats()

    def models_are_different(self, mdl1, mdl2):
        net_distance_l2 = 0
        net_distance_l2 += sum([torch.norm(k1-k2, p=2).item()
                             for k1, k2 in zip(mdl1.parameters(), mdl2.parameters())])
        return net_distance_l2

    def check_models(self):
        distances = []
        print("Check models are different")
        for i in range(len(self._paiv_list)):
            paiv_dist = []
            for j in range(len(self._paiv_list)):
                dist = self.models_are_different(
                    self._paiv_list[i]._model, self._paiv_list[j]._model)
                if self._args.verbose:
                    print(f'Paiv {i}, Paiv {j};{dist}')
                paiv_dist.append(dist)
            distances.append(paiv_dist)
        return distances


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

    def send_local_model(self):
        # the local model is sent to selected destination nodes
        if self._args.verbose:
            print("Sending the local model to:", list(
                self._paiv_list[self._current_node].get_dst_list()))
        for dst in self._paiv_list[self._current_node].get_dst_list():
            self._paiv_list[dst].msg_buffer.put(
                Message(self._current_time, self._current_node, self._paiv_list[self._current_node].model))

    def write_stats(self):

        filename = "stats/" + self._args.outfolder + \
            "/loss_isolation_" + str(self._args.seed) + ".tsv"
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
