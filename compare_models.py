from icecream import ic
from torchvision import datasets, transforms
from models import MLP
import random
import torch
import utils.utils as uutils
import numpy as np
import json
import os
import csv

def get_models_distance(mdl1, mdl2):
        net_distance_l2 = 0
        net_distance_l2 += sum([torch.norm(k1-k2, p=2).item()
                             for k1, k2 in zip(mdl1.parameters(), mdl2.parameters())])
        return net_distance_l2

def main():

    comm_rounds = 40
    seed = 42
    basepath1 = "stats/er_50_02-debug/sensitivity-alpha/fed_diff/distillationSP_vteach-fixed-fed_diff-sentivity_alpha_0.0-seed_42/models/"
    basepath2 = "stats/er_50_02-debug/sensitivity-alpha/fed_diff/distillationDP_vteach-fixed-fed_diff-sentivity_alpha_0.0-seed_42/models/"

    # for t in range(0,comm_rounds):
    #     # filename1 = f'{basepath1}model_{0}_at_{t}_{seed}.pt'
    #     filename1 = f'{basepath1}model_{0}_at_{t}.pt'
    #     filename2 = f'{basepath2}model_{0}_at_{t}_{seed}.pt'

    #     model1 = torch.load(filename1)
    #     model2 = torch.load(filename2)
        
    #     print(models_are_different(model1, model2))

    for i in [0,1]:
        print(f'Differences between the models of node {i} between SP and DP over time')
        for t in range(0,comm_rounds):
            filename1 = f'{basepath1}model_{i}_at_{t}_{seed}.pt'
            # filename1 = f'{basepath1}model_{i}_at_{t}.pt'
            filename2 = f'{basepath2}model_{i}_at_{t}_{seed}.pt'

            model1 = torch.load(filename1)
            model2 = torch.load(filename2)
            
            print(f'Time {t}: beginning of train() -> {get_models_distance(model1, model2)}')

            filename1 = f'{basepath1}model_after_feddiff_{i}_at_{t}_{seed}.pt'
            # filename1 = f'{basepath1}model_{i}_at_{t}.pt'
            filename2 = f'{basepath2}model_after_feddiff_{i}_at_{t}_{seed}.pt'

            model1 = torch.load(filename1)
            model2 = torch.load(filename2)
            
            print(f'Time {t}: after feddiff -> {get_models_distance(model1, model2)}')

            filename1 = f'{basepath1}model_after_retrain_{i}_at_{t}_{seed}.pt'
            # filename1 = f'{basepath1}model_{i}_at_{t}.pt'
            filename2 = f'{basepath2}model_after_retrain_{i}_at_{t}_{seed}.pt'

            model1 = torch.load(filename1)
            model2 = torch.load(filename2)
            
            print(f'Time {t}: after retrain -> {get_models_distance(model1, model2)}')

if __name__ == "__main__":
    main()