#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse
from enum import Enum


class SupportedDatasets(Enum):
    cifar10 = 0
    mnist = 1
    emnist = 2
    fashion_mnist = 3
    cifar100 = 4


def args_parser():
    parser = argparse.ArgumentParser()
    # check point args 

    #args.start_round
    parser.add_argument('--start_round', type=int, default=0,
                        help='check point round')
    #stop_hess_comm
    parser.add_argument('--stop_hess_comm', type=int, default=1000,
                    help='Hessian diagonal communication')
    # hesssian beta
    parser.add_argument('--hessian-beta', type=float, default=1,
                    help="hessian beta function")
    # hessian update type moving_avg
    parser.add_argument('--moving_avg', action='store_true',default=True,
                        help='moving Avg') 

    # file_path
    parser.add_argument('--checkpoint_path', type=str, default='None',
                        help='check point file path')
    # args for scaling weights before training
    parser.add_argument('--scale_weights', action='store_true',default=False,
                        help='restore check point')
    # restore_check_point
    parser.add_argument('--restore_check_point', action='store_true',default=True,
                        help='restore check point')

    # federated arguments
    parser.add_argument('--epochs', type=int, default=10,
                        help="rounds of training")
    # TODO: isn't num_users a duplicate of num_paivs? marked for removal [chiara]
    # parser.add_argument('--num_users', type=int,
    #                    default=100, help="number of PAIVs: K")
    parser.add_argument('--frac', type=float, default=0.1,
                        help="the fraction of clients: C")
    parser.add_argument('--local-ep', type=int, default=5,
                        help="the number of local epochs: E")
    parser.add_argument('--local-distil-ep', type=int, default=5,
                        help="the number of local epochs: E")
    parser.add_argument('--local-bs', type=int, default=10,
                        help="local batch size: B")
    parser.add_argument('--bs', type=int, default=32, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=1.0, help="learning rate deacay factor")
    
    parser.add_argument('--no-early-stop', action='store_true',
                        help='disables the early stopping condition when training')
    parser.add_argument('--optimizer',type=str,choices=['SGD','Adam'],default='SGD',help="Possible options are ['SGD','Adam'] ")
    # SGD
    parser.add_argument('--momentum', type=float, default=0.5,
                        help="SGD momentum (default: 0.5)")
    # ADAM
    parser.add_argument('--weight-decay', type=float, default=1e-3,
                        help="L2 weight-decay (default: 1e-3)")
    

    # DATA related args
    parser.add_argument('--split', type=str, default='user',
                        help="train-test split type, user or sample")
    parser.add_argument('--val-split', type=float, default=0.2,
                        help="train-validation split percentage")
    parser.add_argument('--zipf-alpha', type=float, default=1.6,
                        help="alpha value for zipf-based data partitioning (non iid)")
    parser.add_argument('--dataset', 
                        type=str,
                        default='mnist', 
                        choices=[e.name for e in SupportedDatasets], 
                        help='name of dataset')
    parser.add_argument('--dataset-options', type=str, default='none',
                        help="specify here additional dataset options, e.g., the split for EMNIST")
    
    parser.add_argument('--noniid', action='store_true',
                        help='whether i.i.d or not')
    parser.add_argument('--dirichlet', action='store_true',help='Dirichlet-based data partitioning')
    parser.add_argument('--dirichlet-alpha', type=float, default=1,
                        help="Dirichlet concentration parameter. Default settings for iid partitioning")
    # TODO: merge the ones below/above into a general data assignment approach (e.g., string-based)
    parser.add_argument('--pablo-iid', action='store_true',
                        help='whether to use the data assignment used by pablo (with iid dist)')
    parser.add_argument('--pablo-noniid', action='store_true',
                        help='whether to use the data assignment used by pablo (with non-iid data)')
    parser.add_argument('--communitybased', action='store_true',
                        help='whether to use the data assignment by community')
    parser.add_argument('--fixed-size', type=int, 
                        default=60, help="how many images per nodes (-1 for no limit)")
    parser.add_argument('--force-balance', action='store_true',
                        help='whether to force label balance in train and validation')                     
    

    # TODO: remove the arg below [chiara]
    # parser.add_argument('--num_paivs', type=int,
    #                    default=100, help="number of PAIVs: K")
    # model arguments
    parser.add_argument('--model', type=str, default='mlp',choices=['mlp','cnn'], help='model name [mlp | cnn]')
    parser.add_argument('--num_classes', type=int,
                        default=-1, help="number of classes")
    parser.add_argument('--num-channels', type=int, default=1,
                        help="number of channels of imges")

    # social graph arguments
    parser.add_argument("--graph-from", metavar="graph_from", dest="graph_from", default="synth", type=str,
                        help="method for graph creation (one of: file, preset, synth)")
    parser.add_argument("--graph-file", metavar="edgelist_filename", dest="edgelist_filename", default="", type=str,
                        help="Name of the file containing the edge list")
    parser.add_argument("--graph-synth-type", metavar="graph_synth_type", dest="graph_synth_type", default="barabasi_albert_graph", type=str,
                        help="Type of synthetic graph to be generated")
    parser.add_argument("--graph-synth-args", metavar="graph_synth_args", dest="graph_synth_args", default="10,3,7", type=str,
                        help="String of comma-separated values to pass to nx, in the order specified in the docs")
    parser.add_argument("--graph-preset", metavar="graph_preset", dest="graph_preset", default="karate_club_graph", type=str,
                        help="Load graph from preset datasets in networkX")

    # other arguments
    parser.add_argument('--gpu', type=int, default=0,
                        help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping-rounds', type=int,
                        default=10, help='rounds of early stopping')
    parser.add_argument('--communication-rounds', type=int,
                        default=10, help='rounds of model exchange with friends')

    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')

    parser.add_argument('--run_fed', action='store_true',
                        help='Enables federated training')
    parser.add_argument('--run_cent', action='store_true',
                        help='Enables centralised training')
    parser.add_argument('--run_dec', action='store_true',
                        help='Enables decentralised training')
    parser.add_argument('--run_dec_hess', action='store_true',
                        help='Enables decentralised training')
    

    parser.add_argument('--run_dec_distillation', action='store_true',
                        help='Enables decentralised training with distillation')
    parser.add_argument('--run_isolation', action='store_true',
                        help='Enables training in isolation')
    parser.add_argument('--run_all',
                        action='store_true', help='Enables all training styles (decentralised, federated, centralised')
    
    parser.add_argument('--toggle-virtual-kd',
                        action='store_true', help='Needs --run-dec-distill. Enable the Knowledge distillation with virtual aggregate teacher')
    
    parser.add_argument('--toggle-common-init',
                        action='store_true', help='Needs --run-dec. Forces nodes to start from the same initialization')
    
    parser.add_argument('--toggle-aggregate-first',
                        action='store_true', help='Needs --run-dec. If true, the first step of decentralised training is the model aggregation. I False, first local training and then models aggregation')
    
    parser.add_argument('--aggregation-func', type=str,
                        default='fed_avg', choices = ['fed_avg', 'fed_diff', 'cfa', 'fed_diff_hessian_diag'], help="Options: fed_avg | fed_diff | cfa (SAVAZZI)")
    parser.add_argument('--use-weighted_avg', action='store_true',help='If set, the average model is computed by a weighted average where the weights are like in FedAvg, i.e., size of local data divided by total amount of data in the neightbourhood')
    parser.add_argument('--vteacher_generator', type=str,
                        default='fixed', help="Options: fixed | random")
    parser.add_argument('--skd-beta', type=float,
                        default=0.99, help='if --vteacher-generator == fixed: soft label max value for self-KD | if --vteacher-generator == random: one random soft label picked in the range(beta,.99)')
    parser.add_argument('--kd-alpha', type=float,
                        default=0, help='balance ERM and KD - alpha=0 is equivalent to ERM only')
    parser.add_argument('--include-myself', action='store_true',
                        help='Includes the local model in the average model used in the FedDiff aggregation policy')
    # PAIV type
    parser.add_argument('--paiv-type', type=str,
                        default='SimplePaiv', help="Options: simplePaiv | savazzi2Paiv")
    # benchmark config
    parser.add_argument('--cfa-epsilon', type=float, 
                        default=1, help='set the value of the epsilon parameter of CFA')

    # parser.add_argument('--kd-plateau-tolerance', type=float,
    #                     default=1e-2, help='tolerance for recognizing no additional improvement')
    # parser.add_argument('--kd-plateau-patience', type=float,
    #                     default=1, help='for how many comm. rounds to wait before switching to an aggregation based regime ')
    
    
    parser.add_argument('--oracle-mdl-path', type=str,
                            default = 'None', help="path to the oracle model for distillation")
    # output
    parser.add_argument(
        '--dataset_output_path_prefix', type=str, default='./data',
        help='Prefix for the directory path where datasets will be stored after download.'
    )
    parser.add_argument('--outfolder', type=str,
                        default='stats-meeting', help="prefix of folder")
    parser.add_argument('--exp-name', type=str, default='None', help="Experiment name suffix (it shouldn't be too long)")
    parser.add_argument('--write-every', type=float, 
                        default=-1, help='how often (in comm rounds) the output stats should be written (-1 means only at the end of the simulation)')

    # USED FOR DEBUG
    parser.add_argument('--toggle-output-model',
                        action='store_true', help='Prints the model of node 0 at each comm round')
    parser.add_argument('--toggle-model-dist',
                        action='store_true', help='Prints the distance between models every 5 comm rounds')
    
    
                    
    
    
    

    args = parser.parse_args()
    return args
