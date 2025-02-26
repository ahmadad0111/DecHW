
from training_styles.dec_distillation import DecentralisedDistillation
#from training_styles.dec_distillation_with_oracle import DecentralisedDistillationWithOracle
#from training_styles.dec_virtual_distillation import DecentralisedVirtualDistillation
import utils.options as uo
from graph.sai_graph_generator import SAIGraph
import utils.sampling as usamp
# from torchvision import datasets, transforms
from models import MLP,CNNMnist, CNNCifar, CNNFashion, MnistNet, EMNISTCNN, CNNEMnist, ResNet18
import random
import torch
from training_styles.cent_train import CentralisedLearning
from training_styles.fed_train import FederatedLearning
from training_styles.dec_train import DecentralisedLearning
from training_styles.isolated_training import IsolatedLearning
import numpy as np
import json
import os
from datetime import datetime
import argparse
import csv


def write_args(args):
    """Writes the args of the current run into a json file in the stats directory
    """
    filename = "stats/" + args.outfolder + "/config_" + str(args.seed) + ".json"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

def create_all_models(args, num_users, dataset_train, model_type='mlp'):
    number_of_classes = len(dataset_train.class_to_idx) # extract the number of classes directly from the dataset
    local_models = list()
    print('model type:', model_type)
    for i in range(num_users):
        if model_type=='cnn':
            if args.dataset == 'mnist':
                local_models.append(CNNMnist(args = args,num_classes=number_of_classes).to(args.device))
            elif  args.dataset == 'fashion_mnist':
                # local_models.append(CNNFashion(args=args).to(args.device))
                local_models.append(MnistNet(args=args).to(args.device))
            elif args.dataset == 'cifar10':
                local_models.append(CNNCifar(num_classes=number_of_classes).to(args.device))
            elif args.dataset == 'cifar100':
                #local_models.append(CNNCifar(args=args).to(args.device))
                local_models.append(ResNet18(num_classes=number_of_classes).to(args.device))
            elif args.dataset == 'emnist':
                # local_models.append(MnistNet(args=args).to(args.device))
                # local_models.append(CNNMnist(args=args).to(args.device))
                local_models.append(CNNEMnist(args=args, number_of_classes=number_of_classes).to(args.device))
                # local_models.append(EMNISTCNN(args, 40, 160, 200, 0.4).to(args.device)) # taken from here: https://github.com/austin-hill/EMNIST-CNN
        else: 
            len_in = 1
            for x in dataset_train[0][0].shape:
                len_in *= x
            local_models.append(MLP(dim_in=len_in, dim_out=number_of_classes).to(args.device))
        # set it in training mode
        local_models[i].train()
    return local_models


import argparse

def args_parser():
    parser = argparse.ArgumentParser(description='Training script for various learning styles.')

    parser.add_argument('--noniid', action='store_true', help='Use non-IID data.')
    parser.add_argument('--run_dec', action='store_true', help='Run Decentralised Learning.')
    parser.add_argument('--run_fed', action='store_true', help='Run Federated Learning.')
    parser.add_argument('--run_cent', action='store_true', help='Run Centralised Learning.')
    parser.add_argument('--run_dec_distillation', action='store_true', help='Run Decentralised Distillation.')
    parser.add_argument('--run_isolation', action='store_true', help='Run Isolated Learning.')
    parser.add_argument('--run_all', action='store_true', help='Run all training styles.')
    return parser.parse_args()


def save_checkpoint(args, local_models, round_num, file_path="checkpoint.pth"):
    checkpoint = {
        'round_num': round_num,
        'local_models': [model.state_dict() for model in local_models]
        }

    torch.save(checkpoint, file_path)
    print(f"Checkpoint saved at round {round_num} to {file_path}")


def load_checkpoint(file_path="checkpoint.pth"):
    if os.path.exists(file_path):
        checkpoint = torch.load(file_path)
        print(f"Checkpoint loaded from {file_path}")
        return checkpoint
    else:
        print(f"No checkpoint found at {file_path}")
        return None


def main():
    args = uo.args_parser()  # read args
    write_args(args)  # writing the args for logging

    args.device = torch.device('cuda:{}'.format(
        args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    print(args.device)

    # setting up the social graph based on the input parameters
    g = SAIGraph(args)
    

    # dataset split
    # dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
    #                                transform=transforms.Compose([
    #                                    transforms.ToTensor(),
    #                                    transforms.Normalize(
    #                                        (0.1307,), (0.3081,))
    #                                ]))
    # dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True,
    #                               transform=transforms.Compose([
    #                                   transforms.ToTensor(),
    #                                   transforms.Normalize(
    #                                       (0.1307,), (0.3081,))
    #
    seed = args.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Torch RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Python RNG
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    
    print (f'Main variable: {args.dataset}')
    if args.noniid:
        # data_partitions = usamp.mnist_noniid(
        #     dataset_train, g.sai_graph.number_of_nodes())
        # cls = [i for i in args.num_classes]
        # dataset_train,dataset_test,data_partitions = usamp.mnist_noniid_truncated(args,num_users=len(g.sai_graph.nodes), classes=min(10,g.sai_graph.number_of_nodes()), zipf_alpha=args.zipf_alpha)
        dataset_train,dataset_test,data_partitions = usamp.zipf_noniid_truncated(args,num_users=len(g.sai_graph.nodes), classes=-1, zipf_alpha=args.zipf_alpha)
    elif args.pablo_iid:
        # this is how we set the data when replicating Pablo's experiments
        dataset_train,dataset_test,data_partitions = usamp.dataset_iid_pablo(args, num_users=g.sai_graph.number_of_nodes())
    elif args.pablo_noniid: # DO NOT USE
        # this is how we set the data when replicating Pablo's experiments
        dataset_train,dataset_test,data_partitions = usamp.zipf_noniid_truncated_pablo(args, num_users=g.sai_graph.number_of_nodes())
    elif args.communitybased:
        # here we assign images such that labels are not overlapping across classes
        dataset_train,dataset_test,data_partitions = usamp.dataset_community_based(args, g)
    elif args.dirichlet:
        dataset_train, dataset_test, data_partitions = usamp.dirichlet_dataset_partitioner(
            args, num_users=g.sai_graph.number_of_nodes(), dirichlet_alpha=args.dirichlet_alpha, seed=seed)
    else:
        # dataset_train, dataset_test, data_partitions = usamp.mnist_iid_truncated(
        #     args,g.sai_graph.number_of_nodes())
        dataset_train, dataset_test, data_partitions = usamp.random_iid_truncated(args,num_users=g.sai_graph.number_of_nodes(), classes=min(10, g.sai_graph.number_of_nodes()))
   
    filename = "stats/" + args.outfolder + "/datadist_" + str(args.seed) + ".csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # tmp = dataset_train.class_to_idx.keys()
    if args.num_classes == -1: 
        print(f'Using the all {len(dataset_train.class_to_idx.keys())} classes')
        args.num_classes = dataset_train.classes
    
    with open(filename, 'w') as f:
        wr = csv.writer(f)
        wr.writerow(["nodeid"] + [str(i) for i in dataset_train.class_to_idx.values()])
        # data_partitions_merged = { node: torch.cat((data_partitions['train'][node], data_partitions['validation'][node])) for node in data_partitions['train'].keys()}
        for key,idxs in data_partitions.items():

            # tmp=torch.unique(dataset_train.targets[list(torch.cat((idxs['train'],idxs['validation'])))],return_counts=True)
            labels, counters=torch.unique(dataset_train.targets[list(torch.cat((idxs['train'],idxs['validation'])))],return_counts=True)
            all_labels_at_zero = dict(zip(labels.tolist(), counters.tolist()))
            res = dict.fromkeys(dataset_train.class_to_idx.values(), 0)# | all_labels_at_zero # needed because not all labels are assigned (e.g., EMNIST 'N/A' not in train)
            res.update(all_labels_at_zero)
            # print(len(list(res.values())))
            wr.writerow([key] +  list(res.values()))
    
    
    # ic(args.num_classes)
    run_dec = False
    run_fed = False
    run_cent = False
    run_dec_distil = False
    run_virtual_distil = False # not used anymore
    run_isolation = False
    run_dec_hessian_weighted = False

    if args.run_all:
        run_dec = True
        run_fed = True
        run_cent = True
        run_dec_distil = True
        run_isolation = True
        run_dec_hessian_weighted = True
    else:
        run_dec = args.run_dec
        run_dec_hessian_weighted = args.run_dec_hess
        run_fed = args.run_fed
        run_cent = args.run_cent
        run_dec_distil = args.run_dec_distillation
        run_isolation = args.run_isolation

    if not (run_cent or run_dec or run_fed or run_dec_distil or run_isolation or run_dec_hessian_weighted):
        print('WARNING: No training style selected. Please choose at least one.')

    print(f'run_cent: {run_cent}')
    print(f'run_dec: {run_dec}')
    print(f'run_fed: {run_fed}')
    print(f'run_dec_distil: {run_dec_distil}')
    print(f'run_isolation: {run_isolation}')
    print(f'run_dec_hessian_weighted: {run_dec_hessian_weighted}')
    # run the learning
    print(f'Number of nodes: {g.sai_graph.number_of_nodes()}')

    #print status of run_cent or run_dec or run_fed or run_dec_distil or run_isolation or run_dec_hessian_weighted

    # file path for saving the checkpoint 
    args.checkpoint_path = "stats/" + args.outfolder  + '/seed_'+ str(seed) +'_checkpoint.pth'

    # create the models
    local_models = create_all_models(args, g.sai_graph.number_of_nodes(), dataset_train=dataset_train,model_type=args.model)
    local_hess_diags = [{} for _ in range(g.sai_graph.number_of_nodes())]

    print('Number of local models:', local_models[0])
    
    if args.scale_weights:
        print('Weights are being scaled prior to training')
        for lcl_model in local_models:
            state_dict = lcl_model.state_dict()

            for k in state_dict.keys():
                state_dict[k] *= 50**(0.5) 
            
            lcl_model.load_state_dict(state_dict)
    #save_checkpoint(args, local_models, round_num=10, file_path=checkpoint_path)

    ### load checkpoint if exists
    checkpoint = load_checkpoint(args.checkpoint_path)
    
    if args.restore_check_point and checkpoint:
    # if checkpoint:
        args.start_round = checkpoint['round_num']
        local_models_states = checkpoint['local_models']
        local_hess_diags = checkpoint['local_hessian_diag']
        
        # Example: Load state_dict into models
        for model, state_dict in zip(local_models, local_models_states):
            model.load_state_dict(state_dict)
        print(f"Models restored to state at round {args.start_round}")
    else:
        print("No checkpoint found, starting from scratch")

    print(datetime.now())

    # running the SAI dec-distillation algo
    if run_dec_distil:
        #if args.oracle_mdl_path != "None":
        #    print ('In the Oracle distillation branch')
        #    dec_learn = DecentralisedDistillationWithOracle(
        #        args, g, dataset_train, data_partitions, dataset_test, local_models)
        # elif args.toggle_virtual_kd: 
        #     # DEPRECATED: this was used for the average of neighbor's prob outputs.
        #     dec_learn = DecentralisedVirtualDistillation(
        #         args, g, dataset_train, data_partitions, dataset_test)
        #else:
        dec_learn = DecentralisedDistillation(
            args, g, dataset_train, data_partitions, dataset_test, local_models)
                
        dec_learn.run()

    # running the SAI decentralised algo
    if run_dec:
        dec_learn = DecentralisedLearning(
                args, g, dataset_train, data_partitions, dataset_test, local_models, local_hess_diags, common_init=args.toggle_common_init)
        dec_learn.run()

    # running training in isolation
    if run_isolation:
        isol_learn = IsolatedLearning(
                args, g, dataset_train, data_partitions, dataset_test, local_models)
        isol_learn.run()

    # running federated learning
    if run_fed:
        fed_learn = FederatedLearning(
            args, g, dataset_train, data_partitions, dataset_test, local_models)
        fed_learn.run()

    # # running centralised learning
    if run_cent:
        centr_learn = CentralisedLearning(
            args, dataset_train, data_partitions, dataset_test, local_models)
        centr_learn.run()

    # running the SAI decentralised algo + hessian weighted averaging
    if run_dec_hessian_weighted:
        args.paiv_type = 'hessianPaiv'
        dec_learn = DecentralisedLearning(
                args, g, dataset_train, data_partitions, dataset_test, local_models,local_hess_diags, common_init=args.toggle_common_init)
        dec_learn.run()

    print(datetime.now())


if __name__ == "__main__":
    main()
