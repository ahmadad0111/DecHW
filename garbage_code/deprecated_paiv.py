
# DEPRECATED
class VirtualDistillationPaiv(AbstractPaiv):
    '''
        DEPRECATED
    '''
    def __init__(self, id, args, graph, dataset, data_idxs, model, train_role):
        super().__init__(
            id=id, args=args, graph=graph, dataset=dataset, data_idxs=data_idxs,
            model=model)
        # super(AbstractPaiv,self).__init__(id, args, graph, dataset, data_idxs, model, train_role)

        self._msg_buffer = PriorityQueue()

        # training related properties
        self._loss_local = []
        # self._valid_loss = np.Inf

        # This is used only for test
        self._err_func = nn.CrossEntropyLoss()

        # train_size = int(np.floor(len(data_idxs) * (1-self.args.val_split)))
        # randidx = torch.randperm(len(data_idxs))
        # self.train_idx = data_idxs[randidx[:train_size]]
        # self.val_idx = data_idxs[randidx[train_size:]]
        self.train_idx = data_idxs['train']
        self.val_idx = data_idxs['validation']
        print(
            f'Data distrib: {torch.unique(dataset.targets[self.val_idx],return_counts=True)}')

        self._self_dstil_strategy = SelfDistillationLocalUpdate(
            self._args, dataset=self._dataset, idxs=(self.train_idx, self.val_idx), loss_func=nn.CrossEntropyLoss(), val_loss_func=nn.CrossEntropyLoss())

        self._virtual_KD_strategy = VirtualDistillationLocalUpdate(self._args, dataset=self._dataset, idxs=(
            self.train_idx, self.val_idx), loss_func=nn.CrossEntropyLoss(), val_loss_func=nn.CrossEntropyLoss())

        # create the structure for keeping the label-related probabilies, also consistent with the labels locally present.
        self._local_probs_map, self._local_probs_counts = self._virtual_KD_strategy.get_local_aggregate_soft_labels(
            self.model, init=True)

        self._best_valid_loss = np.Inf

        self._train_role = train_role

        self.plateau = False
        self._patience = 0
        self._learning_delta = np.Inf

        # self._teacher_map = {}

        self._first_train = True

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

    def train(self, current_time):
        # here we implement the atomic training that includes interaction between local and peers' models
        # workflow:
        # 1) process peers' messages
        # 2) train local model embedding peers' knowledge
        if self.args.verbose:
            print("PAIV", self._id, "/ The msg_buffer size is:",
                  len(self._msg_buffer.queue))
        models_out = {}

        active_neighs = list()
        if not self._msg_buffer.empty():
            # note: the [:] is necessary, it creates a copy of
            for msg in self._msg_buffer.queue[:]:
                # self._msg_buffer.queue and the removal with the .get() is applied
                # only when msg.time < current_time
                if msg.time < current_time:
                    if self.args.verbose:
                        print("TRAIN: adding model from", msg.source,
                              "generated at time", msg.time)
                    # associate model with their source node for weighting
                    models_out[msg.source] = msg.outs
                    active_neighs.append(msg.source)
                    self._msg_buffer.get()  # remove the processed message from the message buffer
            if self.args.verbose:
                print("-- ", self._msg_buffer.qsize(),
                      "messages already available for the next training epoch")

        # Perform distillation
        loss = self._self_distillation_with_virtual_aggregate_teacher(
            models_outputs=models_out)

        return loss

    def _self_distillation_with_virtual_aggregate_teacher(self, models_outputs, tolerance=1e-2, patience=2):

        loss = -1
        if self._first_train or not self.plateau:

            # execute self distillation
            # print('[*** Running self-distillation ***]')
            w, loss, vloss = self._self_dstil_strategy.train(
                net=copy.deepcopy(self._model).to(self._args.device))
            self._model.load_state_dict(w)
            self._first_train = False

        else:
            # create the virutal teacher
            # print('[*** Running virtual-KD ***]')
            # virtual_teacher_map = self._compute_virtual_teacher(models_outputs,self._virtual_KD_strategy.unique_labels)
            virtual_teacher_map = self._compute_virtual_teacher_max_entropy(
                models_outputs, self._virtual_KD_strategy.unique_labels)

            # perform the local training though Virtual Distillation Strategy
            w, loss, vloss = self._virtual_KD_strategy.train(net=copy.deepcopy(self._model).to(
                self._args.device), teachers_map=virtual_teacher_map, alpha=self._args.kd_alpha)
            self._model.load_state_dict(w)

        # Obtain the new local aggregate outputs (check if it better using softmax directly or the raw outs)
        self._local_probs_map, self._local_probs_counts = self._virtual_KD_strategy.get_local_aggregate_soft_labels(
            self.model)

        current_delta = self.best_valid_loss-vloss
        if current_delta <= tolerance or current_delta < 0:
            self._patience += 1
        if vloss < self._best_valid_loss:
            self._best_valid_loss = vloss
        if self._patience == patience:
            self.plateau = True
            self._patience = 0

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

                loss = self._err_func(log_probs, labels)
                batch_valid_loss += loss.item()
                batch_accuracy += accuracy.item()

            batch_valid_loss /= batch_idx+1
            batch_accuracy /= batch_idx+1

        return batch_valid_loss, batch_accuracy

    def get_conf_mat(self, dataset, model=None):
        if model is None:
            model = self.model
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
        return confusion_matrix.compute()

    def get_outs(self):
        return self._local_probs_map, self._local_probs_counts

    def _compute_virtual_teacher(self, paivs_outs, unique_labels):
        # this method aggregate the neightbours' output probabilites through a weighted average. The weights are proportional to how many times a given label contributed to forming the current local avg probs.
        nb_classes = len(unique_labels)

        aggregate_probs = {l: torch.zeros(
            (nb_classes,)).to(self.args.device) for l in unique_labels}
        aggr_den = {l: torch.zeros((1,)).to(self.args.device)
                    for l in unique_labels}

        for (probs, counts) in paivs_outs.values():
            for k in probs.keys():
                aggregate_probs[k] += counts[k] * probs[k]
                aggr_den[k] += counts[k]

        for k in self._local_probs_map.keys():
            aggregate_probs[k] = aggregate_probs[k]/aggr_den[k]

        return aggregate_probs

    def _compute_virtual_teacher_max_entropy(self, paivs_outs, unique_labels):
        nb_classes = len(unique_labels)
        aggregate_probs = {l: torch.zeros(
            (nb_classes,)).to(self.args.device) for l in unique_labels}

        current_entropy = {l: [self.id, torch.inf] for l in unique_labels}

        for pid, (probs, _) in paivs_outs.items():
            for k in probs.keys():
                entropy = torch.distributions.Categorical(
                    probs=probs[k]).entropy()
                if entropy < current_entropy[k][1]:
                    aggregate_probs[k] = probs[k]
                    current_entropy[k] = [pid, entropy]

        # print(f'Paiv {self.id} is using v-teachers from paivs: {[val[0] for val in current_entropy.values()]}')
        return aggregate_probs




class OracleDistillationPaiv(AbstractPaiv):
    def __init__(self, id, args, graph, dataset, data_idxs, model, train_role, oracle_model):
        super().__init__(
            id=id, args=args, graph=graph, dataset=dataset, data_idxs=data_idxs,
            model=model)
        # super(AbstractPaiv,self).__init__(id, args, graph, dataset, data_idxs, model, train_role)

        self._msg_buffer = PriorityQueue()

        # training related properties
        self._loss_local = []
        # self._valid_loss = np.Inf

        # This is used only for test
        self._err_func = nn.CrossEntropyLoss()

        # train_size = int(np.floor(len(data_idxs) * (1-self.args.val_split)))
        # randidx = torch.randperm(len(data_idxs))
        # self.train_idx = data_idxs[randidx[:train_size]]
        # self.val_idx = data_idxs[randidx[train_size:]]
        self.train_idx = data_idxs['train']
        self.val_idx = data_idxs['validation']
        print(
            f'Data distrib: {torch.unique(dataset.targets[self.val_idx],return_counts=True)}')

        self._oracle_model = oracle_model

        self._oracle_dstil_strategy = IdealDistillationLocalUpdate(
            self._args, dataset=self._dataset, idxs=(self.train_idx, self.val_idx), loss_func=nn.CrossEntropyLoss(), val_loss_func=nn.CrossEntropyLoss())

        self._best_valid_loss = np.Inf

        self._train_role = train_role

        self.plateau = False
        self._patience = 0
        self._learning_delta = np.Inf

        # self._teacher_map = {}

        self._first_train = True

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

    def train(self, current_time):

        if self.args.verbose:
            print("PAIV", self._id)

        loss = self._distillation_with_oracle(self._oracle_model)
        return loss

    def _distillation_with_oracle(self, oracle_model):
        loss = -1

        print('+++ Training with oracle teacher +++')
        w, loss, vloss = self._oracle_dstil_strategy.train(
            self._model, oracle_model, self.args.kd_alpha)
        

        if vloss < self._best_valid_loss:
            self._best_valid_loss = vloss
            self._model.load_state_dict(w)

        if self.args.verbose:
            print(f'Loss: {loss}; V-Loss: {self._best_valid_loss}')
        # torch._assert(
        #     loss >= 0, 'Error, none of the branches have been traversed')
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

                loss = self._err_func(log_probs, labels)
                batch_valid_loss += loss.item()
                batch_accuracy += accuracy.item()

            batch_valid_loss /= batch_idx+1
            batch_accuracy /= batch_idx+1

        return batch_valid_loss, batch_accuracy

    def get_conf_mat(self, dataset, model=None):
        if model is None:
            model = self.model
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
        return confusion_matrix.compute()



class DistillationPaiv(AbstractPaiv):
    '''
    This is a replica of SimplePaiv. Do not use it. 19/12/2022
    '''
    def __init__(self, id, args, graph, dataset, data_idxs, model, train_role):
        super().__init__(
            id=id, args=args, graph=graph, dataset=dataset, data_idxs=data_idxs,
            model=model)
        # super(AbstractPaiv,self).__init__(id, args, graph, dataset, data_idxs, model, train_role)

        self._msg_buffer = PriorityQueue()

        # training related properties
        self._loss_local = []
        # self._valid_loss = np.Inf
        self._local_update_strategy = []
        # This is used only for test
        self._err_func = nn.CrossEntropyLoss()

        # train_size = int(np.floor(len(data_idxs) * (1-self.args.val_split)))
        # randidx = torch.randperm(len(data_idxs))
        # self.train_idx = data_idxs[randidx[:train_size]]
        # self.val_idx = data_idxs[randidx[train_size:]]
        self.train_idx = data_idxs['train']
        self.val_idx = data_idxs['validation']
        print(
            f'Data distrib: {torch.unique(dataset.targets[self.val_idx],return_counts=True)}')

        self._self_dstil_strategy = SelfDistillationLocalUpdate(
            self._args, dataset=self._dataset, idxs=(self.train_idx, self.val_idx), loss_func=nn.CrossEntropyLoss(), val_loss_func=nn.CrossEntropyLoss())

        self._best_valid_loss = np.Inf

        self._train_role = train_role

        self.aggregation_func = self._args.aggregation_func

        self.toggle_aggregation = self.args.toggle_aggregate_first
        
        self._learning_delta = np.Inf

        # self._teacher_map = {}

        self._first_train = True

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

    def train(self, current_time):
        # here we implement the atomic training that includes interaction between local and peers' models
        # workflow:
        # 1) process peers' models
        # 2) train local model embedding peers' knowledge
        if self.args.verbose:
            print("PAIV", self._id, "/ The msg_buffer size is:",
                  len(self._msg_buffer.queue))

        if self._args.toggle_output_model and self._id in [0,1]:
            filename = f'stats/{self._args.outfolder}/models/model_{self._id}_at_{current_time}_{self._args.seed}.pt'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            print(f'Saving models in {filename}')
            torch.save(self._model, filename)
        
        active_neighs = list()
        suffix = ''
        if not self._msg_buffer.empty():
            # implement logic for selecting the models to aggregate
            # simple: use all models in the buffer
            models = {}
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
                    active_neighs.append(msg.source)
                    self._msg_buffer.get()  # remove the processed message from the message buffer
            if self.args.verbose:
                print("-- ", self._msg_buffer.qsize(),
                      "messages already available for the next training epoch")

        # save the current model
        self._model_previous_round = copy.deepcopy(self._model)
        # Decentralised using Self Distillation for local training abd FedDiffAvg for aggregation
        if active_neighs:
            if self._train_role == 'standalone':
                if self.aggregation_func == 'fed_avg':
                    # aggregate models using social federated average aggregation
                    self._social_aggregation(
                        active_neighs=active_neighs, models=models)
                elif self.aggregation_func == 'fed_diff':
                    self._social_diff_aggregation(
                        active_neighs=active_neighs, models=models)
            
        # DEBUG: we print the models of selected nodes
        if self._args.toggle_output_model and self._id in [0,1]:
            filename = f'stats/{self._args.outfolder}/models/model_after_feddiff_{self._id}_at_{current_time}_{self._args.seed}.pt'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            print(f'Saving models in {filename}')
            torch.save(self._model, filename)
                

        w, loss, vloss = self._self_dstil_strategy.train(
            net=self._model.to(self._args.device))

        # update the local model if the new validation loss improves over the previous one
        if vloss < self._best_valid_loss:
            self._best_valid_loss = vloss
            self._model.load_state_dict(w)
        else:
            # revert to the model before the aggregation + retrain
            self._model = copy.deepcopy(self._model_previous_round)


        # DEBUG: we print the models of selected nodes
        if self._args.toggle_output_model and self._id in [0,1]:
            filename = f'stats/{self._args.outfolder}/models/model_after_retrain_{self._id}_at_{current_time}_{self._args.seed}.pt'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            print(f'Saving models in {filename}')
            torch.save(self._model, filename)
        
        # if vloss < self._best_valid_loss:
        #     self._best_valid_loss = vloss

        return loss


# *** Aggregation functions


    def _selfKD_with_FedAvg(self, models, active_neighs):
        '''
        Local training done using self-distillation
        Aggregation using FedAvg
        '''
        loss = -1

        suffix = ''

        if not self._first_train:
            self._social_aggregation(
                active_neighs=active_neighs, models=models)
            suffix = '-FedAvg'

        w, loss, vloss = self._self_dstil_strategy.train(
            net=self._model, alpha=self.args.kd_alpha)
        self._model.load_state_dict(w)

        self._first_train = False

        self._local_update_strategy.append(f'SelfKD{suffix}')

        if vloss < self._best_valid_loss:
            self._best_valid_loss = vloss

        return loss

    def _selfKD_with_FedDiffAvg(self, models, active_neighs):
        '''
        Local training done using self-distillation
        Aggregation using FedDiff
        '''
        loss = -1

        suffix = ''

        if not self._first_train:
            self._social_diff_aggregation(
                active_neighs=active_neighs, models=models)
            suffix = '-FedDiffAvg'

        w, loss, vloss = self._self_dstil_strategy.train(
            net=self._model)
        self._model.load_state_dict(w)

        self._first_train = False

        self._local_update_strategy.append(f'SelfKD{suffix}')

        if vloss < self._best_valid_loss:
            self._best_valid_loss = vloss

        return loss

    def _self_distillation_with_fallback_on_plateau_DEPRECATED(self, models, active_neighs, tolerance=1e-2, patience=3):
        '''
        DEPRECATED
        '''
        loss = -1

        if not self.plateau:
            print('[*** Running self-distillation ***]')
            w, loss, vloss = self._self_dstil_strategy.train(
                net=copy.deepcopy(self._model).to(self._args.device))
            self._model.load_state_dict(w)
        else:
            print('[*** Running averaging ***]')
            self._social_aggregation(
                active_neighs=active_neighs, models=models)

            w, loss, vloss = self._standard_strategy.train(
                net=copy.deepcopy(self._model).to(self._args.device))

            # TODO: add conditional statement check if the performance degradation is too high.
            self._model.load_state_dict(w)
            # seto to True for switching back to self-distillation after one pass of averaging.
            self.plateau = False

        current_delta = self.best_valid_loss-vloss
        if current_delta <= tolerance or current_delta < 0:
            self._patience += 1
        if vloss < self._best_valid_loss:
            self._best_valid_loss = vloss
        if self._patience == patience:
            self.plateau = True
            self._patience = 0

        return loss

    def _pure_distillation(self, models):
        '''
        Before the first comm. round it uses standard local training (no self-distillation)
        After models' exchange: KD using real models selected according to their per-label performance
        If there are no teachers: fallback to standard local training
        '''
        loss = -1
        whereami = 'out'
        if self._first_train:
            # update local model starting from new aggregate model
            # print('+++ First local training +++')
            w, loss, vloss = self._standard_strategy.train(
                net=self._model)
            self._model.load_state_dict(w)
            self._first_train = False
            # init teachers with local model
            self._distil_strategy.init_teachers(self._model)
            self._local_update_strategy.append('Start-Std')
        else:

            # do the training.
            # if np.random.uniform() > 1-self.args.distill_prob:

            # print('[*** Running distillation ***]')
            teachers = []
            for neight_model in models.values():
                mdl = copy.deepcopy(self._model)
                mdl.load_state_dict(neight_model)
                teachers.append(mdl)
            if len(teachers) > 0:
                # print('+++ Training with teachers +++')
                w, loss, vloss = self._distil_strategy.train(
                    self._model, teachers, alpha=self.args.kd_alpha)
                self._model.load_state_dict(w)
                del teachers
                self._local_update_strategy.append('RealKD')
                # whereami = 'Train with teacher'
            else:
                # print('+++ No teachers: fallback to standard training +++')
                w, loss, vloss = self._standard_strategy.train(self._model)
                self._model.load_state_dict(w)
                # whereami = 'Train without teacher'
                self._local_update_strategy.append('NoTeach-Std')

        if vloss < self._best_valid_loss:
            self._best_valid_loss = vloss
        if self.args.verbose:
            print(f'Loss from "{whereami}": {loss}')
        # torch._assert(
        #     loss >= 0, 'Error, none of the branches have been traversed')
        return loss

    def _distillation_with_fallback_on_plateau(self, models, active_neighs, tolerance=1e-2, patience=3):
        if self._first_train:
            # update local model starting from new aggregate model
            print('+++ First local training +++')
            w, loss, vloss = self._standard_strategy.train(
                net=copy.deepcopy(self._model).to(self._args.device))
            self._model.load_state_dict(w)
            self._first_train = False
            # init teachers with local model
            self._distil_strategy.init_teachers(self._model)

        else:
            # do the training.
            # if np.random.uniform() > 1-self.args.distill_prob:
            if not self.plateau:
                print('[*** Running distillation ***]')
                teachers = []
                for neight_model in models.values():
                    mdl = copy.deepcopy(self._model)
                    mdl.load_state_dict(neight_model)
                    teachers.append(mdl)
                if len(teachers) > 0:
                    print('+++ Training with teachers +++')
                    w, loss, vloss = self._distil_strategy.train(
                        self._model, teachers)
                    self._model.load_state_dict(w)
                    del teachers
                else:
                    print('+++ No teachers: fallback to standard training +++')
                    w, loss, vloss = self._standard_strategy.train(self._model)
                    self._model.load_state_dict(w)
            else:
                print('[*** Running averaging ***]')
                self._social_aggregation(
                    active_neighs=active_neighs, models=models)
                w, loss, vloss = self._standard_strategy.train(
                    net=copy.deepcopy(self._model).to(self._args.device))
                # TODO: add conditional statement check if the performance degradation is too high.
                self._model.load_state_dict(w)
                # comment out for switching back to distillation after one pass of averaging.
                self.plateau = False
        current_delta = self.best_valid_loss-vloss
        if current_delta <= tolerance or current_delta < 0:
            self._patience += 1
        if vloss < self._best_valid_loss:
            self._best_valid_loss = vloss
        if self._patience == patience:
            self.plateau = True
            self._patience = 0

        return loss

    def _realKD_with_FedAvg_selfKD_on_plateau(self, models, active_neighs, tolerance=1e-2, patience=1, aggregate_once_per_plateau=False, social_diff=False):
        '''
        This method performs the following: 
        - First local training: self-KD local training 
        - Do r-KD until a plateau is reached
        - If it reaches a plateau: switch to social aggregation (FedAvg) with self-KD
        '''
        if self._first_train:
            # first local training
            # w, loss, vloss = self._standard_strategy.train(
            #     net=self._model.to(self._args.device))
            w, loss, vloss = self._self_dstil_strategy.train(
                net=self._model, alpha=self.args.kd_alpha)
            self._model.load_state_dict(w)
            self._first_train = False
            # init teachers with local model
            self._distil_strategy.init_teachers(self._model)
            # track down the training phases executed by the paiv
            self._local_update_strategy.append('Start-Std')

        else:
            # do the training.
            if not self.plateau:
                teachers = []
                for neight_model in models.values():
                    mdl = copy.deepcopy(self._model)
                    mdl.load_state_dict(neight_model)
                    teachers.append(mdl)

                # Training with teachers
                w, loss, vloss = self._distil_strategy.train(
                    self._model, teachers, alpha=self.args.kd_alpha)
                self._model.load_state_dict(w)

                # track down the training phases executed by the paiv
                self._local_update_strategy.append('RealKD')
            else:
                # Running averaging
                self._social_aggregation(
                    active_neighs=active_neighs, models=models)

                w, loss, vloss = self._self_dstil_strategy.train(
                    self._model)
                self._model.load_state_dict(w)

                self._local_update_strategy.append('FedAvg-SelfKD')
                if aggregate_once_per_plateau:
                    self.plateau = False

        current_delta = self.best_valid_loss-vloss
        if current_delta <= tolerance or current_delta < 0:
            self._patience += 1
        if vloss < self._best_valid_loss:
            self._best_valid_loss = vloss
        if self._patience == patience:
            self.plateau = True
            self._patience = 0

        return loss

    def _realKD_with_Aggregation_on_plateau(self, models, active_neighs, tolerance=1e-2, patience=1, aggregate_once_per_plateau=False, social_diff=False):
        '''
        This method performs the following: 
        - First local training: Standard (no-*KD) local training 
        - Do r-KD until a plateau is reached
        - If it reaches a plateau: switch to social aggregation (FedAvg) with r-KD
        '''
        if self._first_train:
            # first local training
            w, loss, vloss = self._standard_strategy.train(
                net=self._model.to(self._args.device))
            # w, loss, vloss = self._self_dstil_strategy.train(
            #     net=self._model, alpha=self.args.kd_alpha)
            self._model.load_state_dict(w)
            self._first_train = False
            # init teachers with local model
            self._distil_strategy.init_teachers(self._model)
            # track down the training phases executed by the paiv
            self._local_update_strategy.append('Start-Std')

        else:
            teachers = []
            for neight_model in models.values():
                mdl = copy.deepcopy(self._model)
                mdl.load_state_dict(neight_model)
                teachers.append(mdl)
            # do the training.
            if not self.plateau:

                # Training with teachers
                w, loss, vloss = self._distil_strategy.train(
                    self._model, teachers, alpha=self.args.kd_alpha)
                self._model.load_state_dict(w)

                # track down the training phases executed by the paiv
                self._local_update_strategy.append('RealKD')
            else:
                # Running averaging
                if not social_diff:
                    self._social_aggregation(
                        active_neighs=active_neighs, models=models)
                    self._local_update_strategy.append('FedAvg-RealKD')
                else:
                    self._social_diff_aggregation(
                        active_neighs=active_neighs, models=models)
                    self._local_update_strategy.append('FedDiffAvg-RealKD')

                w, loss, vloss = self._distil_strategy.train(
                    self._model, teachers, alpha=self.args.kd_alpha)
                self._model.load_state_dict(w)

                if aggregate_once_per_plateau:
                    self.plateau = False

        current_delta = self.best_valid_loss-vloss
        if current_delta <= tolerance or current_delta < 0:
            self._patience += 1
        if vloss < self._best_valid_loss:
            self._best_valid_loss = vloss
        if self._patience == patience:
            self.plateau = True
            self._patience = 0

        return loss

    def _realKD_with_aggregation(self, models, active_neighs, social_diff=False):
        '''
        This method performs the following: 
        - First local training: Standard (no-*KD) local training 
        - Do r-KD until a plateau is reached
        - If it reaches a plateau: switch to social aggregation (FedAvg) with r-KD
        '''
        if self._first_train:
            # first local training
            w, loss, vloss = self._standard_strategy.train(
                net=self._model.to(self._args.device))
            # w, loss, vloss = self._self_dstil_strategy.train(
            #     net=self._model, alpha=self.args.kd_alpha)
            self._model.load_state_dict(w)
            self._first_train = False
            # init teachers with local model
            self._distil_strategy.init_teachers(self._model)
            # track down the training phases executed by the paiv
            self._local_update_strategy.append('Start-Std')

        else:
            teachers = []
            for neight_model in models.values():
                mdl = copy.deepcopy(self._model)
                mdl.load_state_dict(neight_model)
                teachers.append(mdl)
            # do the training.

            # Running averaging
            if not social_diff:
                self._social_aggregation(
                    active_neighs=active_neighs, models=models)
                self._local_update_strategy.append('FedAvg-RealKD')
            else:
                self._social_diff_aggregation(
                    active_neighs=active_neighs, models=models)
                self._local_update_strategy.append('FedDiffAvg-RealKD')

            w, loss, vloss = self._distil_strategy.train(
                self._model, teachers, alpha=self.args.kd_alpha)
            self._model.load_state_dict(w)

        # current_delta = self.best_valid_loss-vloss
        # if current_delta <= tolerance or current_delta < 0:
        #     self._patience += 1
        if vloss < self._best_valid_loss:
            self._best_valid_loss = vloss
        # if self._patience == patience:
        #     self.plateau = True
        #     self._patience = 0

        return loss

    def _selfKD_with_FedAvg_selfKD_on_plateau_conditional(self, models, active_neighs, tolerance=1e-2, patience=1, aggregate_once_per_plateau=False, strategy_till_plateau='selfKD', social_diff=True):
        '''
        This method performs the following: 
        - first local training: self-distillitation
        - if there are teachers: do local training otherwise self distillation
        - if a paiv reaches a plateau: switch to social aggregation with local self distillation
        '''
        if self._first_train:
            # First local training
            w, loss, vloss = self._self_dstil_strategy.train(
                net=self._model, alpha=self.args.kd_alpha)
            self._model.load_state_dict(w)
            self._first_train = False
            # init teachers with local model
            # self._distil_strategy.init_teachers(self._model)
            self._local_update_strategy.append('Start-SelfKD')

        else:
            if not self.plateau:
                if not strategy_till_plateau == 'selfKD':
                    w, loss, vloss = self._standard_strategy.train(
                        net=self._model)
                    self._local_update_strategy.append('NP-Std')
                else:
                    w, loss, vloss = self._self_dstil_strategy.train(
                        self._model, alpha=self.args.kd_alpha)
                    self._local_update_strategy.append('NP-SelfKD')
                self._model.load_state_dict(w)

            else:
                # Running averaging
                if not social_diff:
                    self._social_aggregation(
                        active_neighs=active_neighs, models=models)
                else:
                    self._social_diff_aggregation(
                        active_neighs=active_neighs, models=models)

                w, loss, vloss = self._self_dstil_strategy.train(
                    self._model, alpha=self.args.kd_alpha)

                self._model.load_state_dict(w)

                self._local_update_strategy.append('FedAvg-SelfKD')
                if aggregate_once_per_plateau:
                    self.plateau = False

        current_delta = self.best_valid_loss-vloss
        if current_delta <= tolerance or current_delta < 0:
            self._patience += 1
        if vloss < self._best_valid_loss:
            self._best_valid_loss = vloss
        if self._patience == patience:
            self.plateau = True
            self._patience = 0

        return loss

    def _distillation_with_oracle(self, oracle_model):
        loss = -1

        # print('+++ Training with teachers +++')
        w, loss, vloss = self._distil_strategy.train(
            self._model, oracle_model)
        self._model.load_state_dict(w)

        if vloss < self._best_valid_loss:
            self._best_valid_loss = vloss
        if self.args.verbose:
            print(f'Loss: {loss}; V-Loss: {self._best_valid_loss}')
        # torch._assert(
        #     loss >= 0, 'Error, none of the branches have been traversed')
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

                loss = self._err_func(log_probs, labels)
                batch_valid_loss += loss.item()
                batch_accuracy += accuracy.item()

            batch_valid_loss /= batch_idx+1
            batch_accuracy /= batch_idx+1

        return batch_valid_loss, batch_accuracy

    def get_conf_mat(self, dataset, model=None):
        if model is None:
            model = self.model
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
        return confusion_matrix.compute()

    def _social_aggregation(self, active_neighs, models):
        trust_dict = self.get_trust_in_neighs()
        trust_active_dict = {k: trust_dict[k]
                             for k in active_neighs if k in trust_dict}

        # prepare models and trust values for averaging
        w_list = [self._model.state_dict()]
        t_list = [self._selfconfidence]
        for k in models.keys():
            w_list.append(models[k])
            t_list.append(trust_active_dict[k])

        # do the averaging
        model_avg = SocialFedAvg(w_list, t_list)
        self._model.load_state_dict(model_avg)

    def _social_diff_aggregation(self, active_neighs, models):
        trust_dict = self.get_trust_in_neighs()
        trust_active_dict = {k: trust_dict[k]
                             for k in active_neighs if k in trust_dict}

        # prepare models and trust values for averaging
        w_list = []
        t_list = []
        for k in models.keys():
            w_list.append(models[k])
            t_list.append(trust_active_dict[k])

            # do the averaging
        model_update = SocialLocalGlobalDiffUpdate(
            self._model.state_dict(), w_list, t_list)
        self._model.load_state_dict(model_update)

    def models_are_different(self, mdl1, mdl2):
        net_distance_l2 = 0
        net_distance_l2 += sum([torch.norm(k1-k2, p=2).item()
                                for k1, k2 in zip(mdl1.parameters(), mdl2.parameters())])
        return net_distance_l2




