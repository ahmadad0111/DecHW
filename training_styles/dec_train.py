from copy import deepcopy
import math
from models import MLP
from paiv import savazziPaiv, simplePaiv, hessianPaiv
from clock import Clock
from message import Message
from icecream import ic
import utils.utils as uutils
import csv
import os
import torch
from os.path import exists
import numpy as np

class DecentralisedLearning:
    """" Manages the learning process

    This is where the clock and the paivs are located. Also where we keep track of how the learning is going.
    """

    def __init__(self, args, g, dataset_train, data_partitions, dataset_test, local_models, local_hess_diags, common_init = False) -> None:
        self._args = args
        self._graph = g.sai_graph
        self._clock = Clock(list(self._graph.nodes),
                            max_comm_rounds=self._args.communication_rounds, start_round=self._args.start_round)

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

        # self._args.paiv_type = 'hessianPaiv'
        # print("Using", self._args.paiv_type)

        self._last_time_output = 0

        # when the common_init flag is set, the same initial model is loaded on all paivs
        if common_init:
            self.label_outfiles = "_coord"

        for pid in list(self._graph.nodes):
            if not common_init:
                self.label_outfiles = ""
                local_model = local_models[pid]
            else:
                local_model = deepcopy(local_models[0])
            
            if self._args.paiv_type == 'savazzi2Paiv':
                paiv = savazziPaiv(
                    id=pid, args=self._args, graph=self._graph, dataset=dataset_train, data_idxs=data_partitions[pid], model=local_model, train_role='standalone')
            
            elif self._args.paiv_type == 'hessianPaiv':
                hess_diag_ = local_hess_diags[pid]
                paiv = hessianPaiv(
                    id=pid, args=self._args, graph=self._graph, dataset=dataset_train, data_idxs=data_partitions[pid], model=local_model, hess_diag_ = hess_diag_, train_role='standalone')
            else:
                paiv = simplePaiv(
                    id=pid, args=self._args, graph=self._graph, dataset=dataset_train, data_idxs=data_partitions[pid], model=local_model, train_role='standalone')
                
            self._paiv_list.append(paiv)
            self._train_loss[pid] = []
            self._validation_loss[pid] = []
            self._test_loss[pid] = []
            self._test_accuracy[pid] = []
        
        self._dataset_test = dataset_test
        # DEBUG: print the existing PAIVs
        if self._args.verbose:
            print("DEC_TRAIN: We have ", len(self._paiv_list), "PAIVs")
            for p in self._paiv_list:
                print("\tI'm paiv", p.id, "and my local data contains",
                      len(p.data_idxs['train']), "items (TRAIN) and ", len(p.data_idxs['validation']), "items (VALIDATION)")

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
        print_counter = 1
        # do until there are not events in the clock
        while not self._clock.empty():
            (self._current_time, self._current_node, self._event_type,
             self._event_counter) = self._clock.get_next()
            #print("Time:", self._current_time, " - serving node", self._current_node)
            
            if self._args.toggle_model_dist and self._current_time % 5 == 0:
            # if self._current_time % 5 >= 0:
                filename = f'stats/{self._args.outfolder}/models_dec{self.label_outfiles}_at_{self._current_time}.pt'
                if not exists(filename):
                    dist_table = self.check_models()

                    print(f'Saving models distances in {filename}')
                    torch.save(torch.tensor(dist_table), filename)

            # checkpoint
            if self._paiv_list[self._current_node].id != self._current_node:
                print("ERROR: the id of the paiv does not correspond to its position in the paiv list [luigi's]")
                exit(-1)
            
            # train
            paiv_loss_train = self._paiv_list[self._current_node].train(
                self._current_time)
            # assert(not math.isnan(paiv_loss_train))
            self._train_loss[self._current_node].append(paiv_loss_train)
            # TODO: swap below, used only for testing while waiting for the validation training
            self._validation_loss[self._current_node].append(
                self._paiv_list[self._current_node].validate())

            tloss, tacc = self._paiv_list[self._current_node].test(
                self._dataset_test)
            self._test_loss[self._current_node].append(tloss)
            self._test_accuracy[self._current_node].append(tacc)
            if self._args.verbose:
                print("Done with training, sending models to neighbors")

            if self._current_time % 1 == 0:
                if print_counter % len(self._train_loss) == 0:
                    print("Time:", self._current_time)
                    print(
                        f'Avg Train Loss at {self._current_time}: {np.mean(np.array([tloss[-1] for tloss in self._train_loss.values()])):.2f}')
                    print(
                        f'Avg Test Loss at {self._current_time}: {np.mean(np.array([tloss[-1] for tloss in self._test_loss.values() ])):.2f}')
                    print(
                        f'Avg Accuracy at {self._current_time}: {np.mean(np.array([tloss[-1] for tloss in self._test_accuracy.values() ])):.2f}')
                    print_counter = 0

                    # checkpoint 
                    # list of all models dicts
                    local_models = [paiv._model for paiv in self._paiv_list]
                    current_round = self._current_time + 1
                    
                    # store checkpoint after every 5 rounds
                    # saving hessian info
                    if self._args.paiv_type == 'hessianPaiv':
                        local_hess_diags = [paiv.hessian_diag for paiv in self._paiv_list]
                        self.save_checkpoint1(local_models, current_round, file_path= self._args.checkpoint_path, local_hess_diags=local_hess_diags)
                    
                    
                    
                    if current_round % 5 == 0:
                        # print("Saving checkpoint at round", current_round)
                        # if self._args.paiv_type == 'hessianPaiv':
                        #     local_hess_diags = [paiv.hessian_diag for paiv in self._paiv_list]
                        #     #self.save_checkpoint(local_models, current_round, file_path= self._args.checkpoint_path, local_hess_diags=local_hess_diags)
                            
                        # else:
                        #     local_hess_diags = [{} for paiv in self._paiv_list]  
                        #     # save at every round
                        #     self.save_checkpoint(local_models, current_round, file_path= self._args.checkpoint_path, local_hess_diags=local_hess_diags)
                        
                        # saving sates
                        self.write_stats_periodically()


                    # lr rate decay every round
                    for paiv in self._paiv_list:
                        paiv._local_strategy.args.lr = paiv._local_strategy.args.lr * self._args.lr_decay
                        break


            print_counter += 1
            # the resulting local model is sent to selected destination nodes
            if self._args.paiv_type == 'savazzi2Paiv':
                self.send_local_model_and_gradients()
            elif self._args.paiv_type == 'hessianPaiv':
                self.send_local_model_and_hessian_diag()
            else:
                self.send_local_model()

            # save check points 

            # os.makedirs(self.args.checkpoint_path, exist_ok=True)


            # print("Current Node is", self._current_node)
            # print("current node model", self._paiv_list[self._current_node]._model.state_dict())



            # every x communication rounds, write to file
            if self._current_time > 0 and self._args.write_every > 0 and self._current_time is not self._clock.time_of_next() and self._current_time % self._args.write_every == 0:
                self.write_stats_periodically()

            self.stop_if_condition_met()





        # print final train loss
        ic(self._train_loss)
        if self._args.write_every > 0:
            self.write_stats_periodically()
        else:
            self.write_stats()

    def save_checkpoint1(self,  local_models,round_num, file_path="checkpoint.pth", local_hess_diags=None ):
        if self._args.paiv_type == 'hessianPaiv':
            #file_path = "round_" + str(round_num) + "_checkpoint.pth"
            #file_path = f'stats/{self._args.outfolder}/round_{round_num}_checkpoint.pth'   
            file_path = "stats/" + self._args.outfolder  + '/round_'+ str(round_num) + '_checkpoint.pth'

            checkpoint = {
            'round_num': round_num,
            'local_hessian_diag': [hess_diag for hess_diag in local_hess_diags]
            }
        else:
            checkpoint = {
                'round_num': round_num,
                'local_models': [model.state_dict() for model in local_models],
                'local_hessian_diag': [hess_diag for hess_diag in local_hess_diags]
                }
        print('hessian is saved to: {}'.format(file_path))
        torch.save(checkpoint, file_path)
    def save_checkpoint(self,  local_models,round_num, file_path="checkpoint.pth", local_hess_diags=None ):
        if self._args.paiv_type == 'hessianPaiv':

            checkpoint = {
            'round_num': round_num,
            'local_models': [model.state_dict() for model in local_models],
            'local_hessian_diag': [hess_diag for hess_diag in local_hess_diags]
            }
        else:
            checkpoint = {
                'round_num': round_num,
                'local_models': [model.state_dict() for model in local_models],
                'local_hessian_diag': [hess_diag for hess_diag in local_hess_diags]
                }

        torch.save(checkpoint, file_path)

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
                # print(
                #     f'Paiv {i}, Paiv {j};{dist}')
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
                Message(self._current_time, self._current_node, self._paiv_list[self._current_node].model, data_size=self._paiv_list[self._current_node].train_size))
            
    # send the local model and the hessian information to the neighbors
    def  send_local_model_and_hessian_diag(self):
        # the local model is sent to selected destination nodes
        if self._args.verbose:
            print("Sending the local model to:", list(
                self._paiv_list[self._current_node].get_dst_list()))
        for dst in self._paiv_list[self._current_node].get_dst_list():
            # if dst in self._paiv_list[self._current_node].neigh_hessian_diag.keys():
            #     hessian_diag = self._paiv_list[self._current_node].neigh_hessian_diag[dst]
            # else:
            #     hessian_diag  = None
            self._paiv_list[dst].msg_buffer.put(
                Message(self._current_time, self._current_node, self._paiv_list[self._current_node].model, data_size=self._paiv_list[self._current_node].train_size, hessian_diag=self._paiv_list[self._current_node].hessian_diag))
            
    def  send_local_model_and_gradients(self):
        # the local model is sent to selected destination nodes
        if self._args.verbose:
            print("Sending the local model to:", list(
                self._paiv_list[self._current_node].get_dst_list()))
        for dst in self._paiv_list[self._current_node].get_dst_list():
            if dst in self._paiv_list[self._current_node].neigh_gradients.keys():
                grad = self._paiv_list[self._current_node].neigh_gradients[dst]
            else:
                grad  = None
            self._paiv_list[dst].msg_buffer.put(
                Message(self._current_time, self._current_node, self._paiv_list[self._current_node].psi_model, data_size=self._paiv_list[self._current_node].train_size, gradients=grad))

    def write_stats(self):

        filename = "stats/" + self._args.outfolder + \
            "/loss_dec" + self.label_outfiles + "_" + str(self._args.seed) + ".tsv"
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
            f.close()

    def write_stats_periodically(self):
        filename = "stats/" + self._args.outfolder + \
                "/loss_dec" + self.label_outfiles + "_" + str(self._args.seed) + ".tsv"
        if self._current_time == 0:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            open_mode = 'w'
        else:
            open_mode = 'a'
        
        with open(filename, open_mode) as f:
            wr = csv.writer(f)
            if self._last_time_output == 0:
                wr.writerow(['nodeid', 'time', 'loss', 'loss_type'])
            for k in self._train_loss.keys():
                for t in range(self._last_time_output, len(self._train_loss[k])):
                    wr.writerow([k, t, self._train_loss[k][t], 'train'])

            for k in self._validation_loss.keys():
                for t in range(self._last_time_output, len(self._validation_loss[k])):
                    wr.writerow(
                        [k, t, self._validation_loss[k][t], 'validation'])

            for k in self._test_loss.keys():
                for t in range(self._last_time_output, len(self._test_loss[k])):
                    wr.writerow([k, t, self._test_loss[k][t], 'test'])

            for k in self._test_accuracy.keys():
                for t in range(self._last_time_output, len(self._test_accuracy[k])):
                    wr.writerow([k, t, self._test_accuracy[k][t], 'accuracy'])
            f.close()
        
        ### DISCLAIMER: this works with integer time
        self._last_time_output = self._current_time+1
        
