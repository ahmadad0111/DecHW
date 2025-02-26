#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        if "num_batches_tracked" not in k:
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def WeightedFedAvg(w,alphas):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        if "num_batches_tracked" not in k:
            w_avg[k] *= alphas[0]

    for k in w_avg.keys():
        if "num_batches_tracked" not in k:
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]*alphas[i]
    return w_avg


# socially weighted federated averaging
def SocialFedAvg(models: list, trust: list, alpha_list: list):
    '''
    Computes the weighted federated averaged of the models
    received from the paiv's social graph, weighted by their social trust
    '''

    w_avg = copy.deepcopy(models[0])
    for k in w_avg.keys():
        if "num_batches_tracked" not in k:
            w_avg[k] *= trust[0]*alpha_list[0]

    for k in w_avg.keys():
        if "num_batches_tracked" not in k:
            for i in range(1, len(models)):
                w_avg[k] += models[i][k]*trust[i]*alpha_list[i]
            # w_avg[k] = torch.div(w_avg[k], sum(trust))
            if sum(trust) <=0:
                if w_avg[k] <= 0:
                    print("Total trust:", sum(trust), " - Total weights:", w_avg[k])
                else:
                    print("* Total trust:", sum(trust), " - Total weights:", w_avg[k])

    return w_avg

# socially weighted federated averaging
def SocialLocalGlobalDiffUpdate(local_model, models: list, trust: list,alphas:list, p_norm=2):
    '''
    Computes the weighted federated averaged of the models
    received from the paiv's social graph, weighted by their social trust
    '''
        
    if len(models) > 0:        
        w_avg = copy.deepcopy(models[0])
        for k in w_avg.keys():
            if "num_batches_tracked" not in k:
                w_avg[k] *= trust[0]*alphas[0]

        for k in w_avg.keys():
            if "num_batches_tracked" not in k:
                for i in range(1, len(models)):
                    w_avg[k] += models[i][k]*trust[i]*alphas[i]

        
        # the update goes in the opposite direction w.r.t. the average model
        for k in w_avg.keys():
            if "num_batches_tracked" not in k:
                # FedDiff update rule: w_local  = w_local - (w_local - w_avg)/||w_local - w_avg||_2
                dist = local_model[k]-w_avg[k]
                lp_dist = torch.norm(dist, p=p_norm)+1
                local_model[k] = local_model[k] - (dist)/(lp_dist)
    
    return local_model


# socially weighted federated averaging with hessian diagonal weighted
def SocialLocalGlobalDiffUpdate_hessian_diag(hessian_terms, local_model, models: list, trust: list,alphas:list, p_norm=2):
    '''
    Computes the weighted federated averaged of the models
    received from the paiv's social graph, weighted by their social trust
    '''
        
    if len(models) > 0:        
        w_avg = copy.deepcopy(models[0])
        for k in w_avg.keys():
            if "num_batches_tracked" not in k:
                if "running_mean" in k or "running_var" in k:
                    w_avg[k] *= trust[0]*alphas[0]
                else:
                    w_avg[k] *= trust[0]*hessian_terms[0][k]#*alphas[0]

        for k in w_avg.keys():
            for i in range(1, len(models)):
                if "running_mean" in k or "running_var" in k:
                    w_avg[k] += models[i][k]*trust[i]*alphas[i]
                elif "num_batches_tracked" in k:
                    w_avg[k] += models[i][k]
                else:
                    w_avg[k] += models[i][k]*trust[i]*hessian_terms[i][k] #*alphas[i]
            # w_avg[k] = torch.div(w_avg[k], sum(trust))
        # # the update goes in the opposite direction w.r.t. the average model
        # for k in w_avg.keys():
        #     if "num_batches_tracked" not in k:
        #         # FedDiff update rule: w_local  = w_local - (w_local - w_avg)/||w_local - w_avg||_2
        #         dist = local_model[k]-w_avg[k]
        #         lp_dist = torch.norm(dist, p=p_norm)+1
        #         local_model[k] = local_model[k] - (dist)/(lp_dist)
    return w_avg



def consensus_federated_average(local_model, models:list,alphas:list,epsilon: float):
    '''
    Computes the Consensus based Federated Average as defined in Algorithm 1 of:
    "Savazzi, Stefano, Monica Nicoli, and Vittorio Rampa. ‘Federated Learning with Cooperating Devices: A Consensus Approach for Massive IoT Networks’. ArXiv 7, no. 5 (2019): 4641–54."
    '''

    temp_model = copy.deepcopy(local_model)
    
    for k in temp_model.keys():
        temp_model[k].zero_()
    
    if len(models)>0:
        for k in local_model.keys():
            for i in range(0,len(models)):
                if "num_batches_tracked" in k:
                   temp_model[k] = temp_model[k]
                else:
                    temp_model[k] += alphas[i]*(models[i][k]-local_model[k])
    
    for k in local_model.keys():
        if "num_batches_tracked" in k:
            temp_model[k] = temp_model[k]
        else:
            local_model[k] += epsilon*temp_model[k]
         

    return local_model
