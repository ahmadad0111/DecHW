#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import logging

from utils.options import args_parser, SupportedDatasets
import numpy as np
from torchvision.datasets import MNIST, EMNIST, FashionMNIST, CIFAR10,CIFAR100
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision import datasets, transforms
import torch
from fedlab.utils.dataset import MNISTPartitioner,FMNISTPartitioner, CIFAR10Partitioner, CIFAR100Partitioner
from fedlab.utils.functional import partition_report

from collections import defaultdict
import random



_MNIST_TRANSFORM = Compose([
    ToTensor(),
    Normalize((0.1307,), (0.3081,))
])
_FASHION_MNIST_TRANSFORM = _MNIST_TRANSFORM
_EMNIST_TRANSFORM = _MNIST_TRANSFORM
_CIFAR10_TRASFORM = Compose([
    ToTensor()
])


def _patch_torchvision_emnist_urls():
    """https://github.com/pytorch/vision/blob/5181a854d8b127cf465cd22a67c1b5aaf6ccae05/torchvision/datasets/mnist.py#L279
    """
    updated_url = 'https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip'
    if EMNIST.url != updated_url:
        EMNIST.url = updated_url
        logging.warning(
            'Found unexpected url for emnist dataset, this is due to torchvision <= v0.17.2. '
            'Patching the url with "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"'
        )


_patch_torchvision_emnist_urls()


def get_dataset(args):
    print(f'Selected: {args.dataset}')
    dataset_output_path = os.path.join(args.dataset_output_path_prefix, args.dataset)

    dataset = SupportedDatasets[args.dataset]

    if dataset == SupportedDatasets.mnist:
        trainset = MNIST(dataset_output_path, train=True, download=True, transform=_MNIST_TRANSFORM)
        testset = MNIST(dataset_output_path, train=False, download=True, transform=_MNIST_TRANSFORM)
    elif dataset == SupportedDatasets.emnist:
        if args.dataset_options == 'none':
            args.dataset_options = 'letters'

        trainset = EMNIST(
            dataset_output_path, 
            split=args.dataset_options, 
            train=True, 
            download=True,
            transform=_EMNIST_TRANSFORM
        )
        testset = EMNIST(
            dataset_output_path, 
            split=args.dataset_options, 
            train=False, 
            download=True,
            transform=_EMNIST_TRANSFORM
        )
    elif dataset == SupportedDatasets.fashion_mnist:
        trainset = FashionMNIST(
            dataset_output_path, 
            train=True, 
            download=True, 
            transform=_FASHION_MNIST_TRANSFORM
        )
        testset = FashionMNIST(
            dataset_output_path, 
            train=False, 
            download=True, 
            transform=_FASHION_MNIST_TRANSFORM
        )
    elif dataset == SupportedDatasets.cifar10:
        trainset = CIFAR10(dataset_output_path, train=True, download=True, transform=_CIFAR10_TRASFORM)
        testset = CIFAR10(dataset_output_path, train=False, download=True, transform=_CIFAR10_TRASFORM)
    elif dataset == SupportedDatasets.cifar100:
        trainset = CIFAR100(dataset_output_path, train=True, download=True, transform=_CIFAR10_TRASFORM)
        testset = CIFAR100(dataset_output_path, train=False, download=True, transform=_CIFAR10_TRASFORM)


    return trainset, testset


# def get_mnist():
#     dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
#                                    transform=transforms.Compose([
#                                        transforms.ToTensor(),
#                                        transforms.Normalize(
#                                            (0.1307,), (0.3081,))
#                                    ]))
#     dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True,
#                                   transform=transforms.Compose([
#                                       transforms.ToTensor(),
#                                       transforms.Normalize(
#                                           (0.1307,), (0.3081,))
#                                   ]))
#     return dataset_train, dataset_test


# def get_fashion():
#     dataset_train = datasets.FashionMNIST(
#         '../data/fashion/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
#     dataset_test = datasets.FashionMNIST(
#         '../data/fashion/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
#     return dataset_train, dataset_test

# DEPRECATED: mnist_iid_pablo now can deal with both limited/non-limited size
def random_iid(args, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    dataset,testset = get_dataset(args)
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]

    for i in range(num_users):
        tmp = set(np.random.choice(
            all_idxs, num_items, replace=False))
        dict_users[i] = torch.tensor(list(tmp))
        all_idxs = list(set(all_idxs) - tmp)

    return dataset,testset,dict_users

def random_iid_truncated(args,num_users, classes=10):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    dataset, testset = get_dataset(args)
    idxs = []
    for i in range(len(dataset.targets)):
        for c in range(classes):
            if dataset.targets[i] == c:
                idxs.append(i)
    dataset.data = dataset.data[idxs]
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        if not isinstance(dataset.targets, torch.Tensor):
            dataset.targets = torch.tensor(dataset.targets)

    dataset.targets = dataset.targets[idxs]

    num_items = int(len(dataset)/num_users)
    dict_users = {}
    all_idxs = {}, [i for i in range(len(dataset))]

    for i in range(num_users):
        dict_users[i] = set(np.random.choice(
            all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    args.num_classes = classes
    return dataset, testset, dict_users

# comment ad cazzum

def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 12000, 5
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    p = np.random.zipf(2.15, num_users)
    prob = p/np.sum(p)
    # divide and assign
    user_sequence = np.random.choice(
        range(num_users), num_shards, p=prob, replace=True)
    while len(set(idx_shard)):
        for i in user_sequence:
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    for uid, idxs in dict_users.items():
        print('id={}, len={}, labels={}'.format(
            uid, len(idxs), np.unique(np.array(labels[idxs]), return_counts=True)))
    return dict_users

def zipf_noniid_truncated_wrapper(args, num_users=10, classes=10, zipf_alpha=1.6, prob_vec=None):
    new_data_partitions = dict()
    train_idx_users = {node: [] for node in range(num_users)} # initialize dict with one empty list per user
    val_idx_users = {node: [] for node in range(num_users)} # initialize dict with one empty list per user

    # we generate the usual distribution as when nodes are >=10
    dataset_train,dataset_test,data_partitions_10 = zipf_noniid_truncated(args, num_users=max(10,num_users), classes=10, zipf_alpha=args.zipf_alpha, prob_vec=prob_vec)

    data_partitions = dict()
    if num_users < classes:
        for i in range(num_users):
            data_partitions[i] = data_partitions_10[i]
    
    samples_per_user = int(len(dataset_train)/num_users) # we use the whole dataset here
    train_size = int(np.floor(samples_per_user * (1-args.val_split)))

    for i, idxs in data_partitions.items():
        print(f'Node {i} has {idxs.size()} indices assigned')
        randidx = torch.randperm(len(idxs))
        print(randidx.size())
        train_idx_users[i] = idxs[randidx[:train_size]]
        val_idx_users[i] = idxs[randidx[train_size:]]
        new_data_partitions[i] = {'train': train_idx_users[i], 'validation' : val_idx_users[i]}

    print_count_per_class(dataset_train, train_idx_users, 'train')
    print_count_per_class(dataset_train, val_idx_users, 'validation')

    return dataset_train,dataset_test,new_data_partitions
    


def zipf_noniid_truncated(args, num_users=10, classes=10, zipf_alpha=1.6, prob_vec=None):
    """
    Sample non-I.I.D client data from {args.dataset} dataset on a subset of classes
    :param dataset:
    :param num_users:
    :param classes:
    :param zipf_alpha:
    :param prob_vec:
    :return:
    """

    new_data_partitions = dict()
    train_idx_users = {node: [] for node in range(num_users)} # initialize dict with one empty list per user
    val_idx_users = {node: [] for node in range(num_users)} # initialize dict with one empty list per user

    dataset,testset = get_dataset(args)

    print(f'The labels for the selected dataset are:\n{testset.class_to_idx}')

    if classes == -1:
        classes = len(testset.class_to_idx)

    # Subsample dataset
    if classes < len(testset.class_to_idx):
        idxs = []
        for i in range(len(dataset.targets)):
            for c in range(classes):
                if dataset.targets[i] == c:
                    idxs.append(i)

        dataset.data = dataset.data[idxs]
        
        dataset.targets = dataset.targets[idxs]
        # Subsample MNIST dataset
        del idxs
        idxs = []
        for i in range(len(testset.targets)):
            for c in range(classes):
                if testset.targets[i] == c:
                    idxs.append(i)

        testset.data = testset.data[idxs]
        testset.targets = testset.targets[idxs]

    if zipf_alpha is not None:
        # we generate a zipfs sample but we keep it only if its sum is <= than the number of images in the least numerous class
        # (otherwise we cannot guarantee that each node has got >=1 image for each class)
        # TO DO: check what is the minimum allowed
        if args.dataset == 'cifar10' or args.dataset == 'cifar100':
            if not isinstance(dataset.targets, torch.Tensor):
                dataset.targets = torch.tensor(dataset.targets)
            items_per_class = [torch.where(dataset.targets == c)[0].shape[0] for c in range(classes)]
        else:
            items_per_class = [torch.where(dataset.targets == c)[0].shape[0] for c in range(classes)]
        print(f'items per class: {items_per_class}')
        print(f'dataset:{args.dataset} dataset option={args.dataset_options}')
        if args.dataset == "emnist" and args.dataset_options == "letters":
            items_per_class.pop(0)
            print("Got rid of class N/A")
        count_discarded = 0
        while True:
            zipfs_samples = np.random.zipf(zipf_alpha, num_users)
            if np.sum(zipfs_samples) <= min(items_per_class):
                print("Zipfs: distribution OK")
                break
            else:
                count_discarded = count_discarded +1 
                print("Checking Zipf: ", count_discarded)
        p = torch.tensor(zipfs_samples)
    else:
        p = torch.tensor(prob_vec)
    probs = p/torch.sum(p)


    partitions = {}
    for j, c in enumerate(np.arange(classes)):
        # print(f'j={j} and c={c}') # j is the current node (?), c is the current class
        i = 0
        # print(f'adding class {c}')
        cidx = torch.where(dataset.targets == c)[0]
        user_probs = torch.roll(probs, j) # torch.roll shifts the probs tensor to the right
        for ix, p in enumerate(user_probs):
            # print(f'adding indices for user {ix} for class {c} with size {p}')
            if ix != len(probs)-1:
                offset = int(np.floor(cidx.shape[0] * p))
                if j == 0:
                    # print(f'init for user {ix}')
                    partitions[ix] = cidx[i:i+offset]
                else:
                    partitions[ix] = torch.hstack(
                        (partitions[ix], cidx[i:i+offset]))
            else:
                last = (cidx.shape[0]-i)
                if j == 0:
                    # print(f'init for user {ix}')
                    partitions[ix] = cidx[-last:]
                else:
                    partitions[ix] = torch.hstack(
                        (partitions[ix], cidx[-last:]))

            i = i+offset
    args.num_classes = classes

    # split into train and validation
    # samples_per_user = int(len(dataset)/num_users) # we use the whole dataset here
    # train_size = int(np.floor(samples_per_user * (1-args.val_split)))

    print_count_per_class(dataset, partitions, 'train+valid')

    for i, idxs in partitions.items():
        # print(f'Node {i} has {len(idxs)} indices assigned')
        train_size = int(np.floor(len(idxs)) * (1-args.val_split))
        randidx = torch.randperm(len(idxs))
        # print(randidx.size())
        train_idx_users[i] = idxs[randidx[:train_size]]
        val_idx_users[i] = idxs[randidx[train_size:]]
        new_data_partitions[i] = {'train': train_idx_users[i], 'validation' : val_idx_users[i]}

    # print_count_per_class(dataset, train_idx_users, 'train')
    # print_count_per_class(dataset, val_idx_users, 'validation')

    return dataset,testset,new_data_partitions

def rebalance_classes(current_assignment):
    # TODO: implement robin-hood policy
    return current_assignment


def downsampleHRHare(tot_local_images_per_class, samples_per_user, force_balance=True):
    # print(f"Downsampling")
    tot_local_images = torch.sum(tot_local_images_per_class)
    hare_quota = tot_local_images/samples_per_user
    votes_per_quota = torch.div(tot_local_images_per_class,hare_quota)
    # print(votes_per_quota)
    automatic_seats = torch.floor(votes_per_quota)
    # print(automatic_seats)
    remainder = torch.remainder(votes_per_quota,automatic_seats)
    # print(remainder)
    left_to_assign = samples_per_user - torch.sum(automatic_seats).item()
    while(left_to_assign > 0):
        max_remainder = torch.max(remainder)
        # print(max_remainder)
        idx_max = (remainder == max_remainder).nonzero(as_tuple=True)[0]
        automatic_seats[idx_max] += 1
        remainder[idx_max] = 0
        left_to_assign -= 1
        # print(automatic_seats)
        # print(remainder)
    if  force_balance:
        automatic_seats = rebalance_classes(automatic_seats)
    return automatic_seats


def zipf_noniid_truncated_pablo(args, num_users=10, prob_vec=None):
    # TODO: check that args.fixed_size is not smaller than 20

    # we generate the usual distribution as when nodes are >=10
    dataset_train,dataset_test,data_partitions = zipf_noniid_truncated(args, num_users=max(10,num_users), classes=10, zipf_alpha=args.zipf_alpha, prob_vec=prob_vec)

    new_data_partitions = dict()
    train_idx_users = {node: [] for node in range(num_users)} # initialize dict with one empty list per user
    val_idx_users = {node: [] for node in range(num_users)} # initialize dict with one empty list per user

    samples_per_user = args.fixed_size # number of items per users
    if samples_per_user == -1:
        samples_per_user = int(len(dataset_train)/num_users) # we use the whole dataset here
    train_size = int(np.floor(samples_per_user * (1-args.val_split)))

    downsampled_partitions = dict()

    # largest remainder method (with hare quota)
    for node,idxs in data_partitions.items():
        if node >= num_users:
            break
        # print(f"the node is {node}")
        labels_with_idx = get_idxs_per_class(dataset_train)
     
        # print(labels_with_idx)
        # print(labels_with_idx.keys()) #prints keys
        
        tot_local_images_per_class = torch.unique(dataset_train.targets[list(idxs)],return_counts=True)[1]
        tot_local_images_per_class_downsampled = downsampleHRHare(tot_local_images_per_class, samples_per_user).tolist()

        labels = list(labels_with_idx.keys())
        tot_local_images_per_class_downsampled_dict = {labels[i]: int(tot_local_images_per_class_downsampled[i]) for i in range(len(tot_local_images_per_class_downsampled))}

        all_idxs_downsampled = list()
        for label, idx_list in labels_with_idx.items():
            all_idxs_downsampled.extend(random.sample(idx_list, tot_local_images_per_class_downsampled_dict[label]))

        downsampled_partitions[node]=torch.LongTensor(all_idxs_downsampled)
        
    # # print the final label count on each node [debug]
    # for key,val in downsampled_partitions.items():
    #     print(f"Label count on node {key}: ")
    #     print(torch.unique(dataset_train.targets[list(val)],return_counts=True))

    for i in downsampled_partitions:
        train_idx_users[i] = torch.tensor(downsampled_partitions[i][:train_size])
        val_idx_users[i] = torch.tensor(downsampled_partitions[i][train_size:])
        new_data_partitions[i] = {'train': train_idx_users[i], 'validation' : val_idx_users[i]}

    return dataset_train,dataset_test,new_data_partitions


def get_idxs_per_class(dataset):
    labels_with_idx = defaultdict(list) # this will contain {class1 : list_of_idxs, class2 : list_of_idxs}
    idxs = [i for i in range(len(dataset))]
    for idx in idxs:
        # print(f"idx is {idx}")
        # print(f"the label is {dataset_train.targets[idx].item()}")
        labels_with_idx[dataset.targets[idx].item()].append(idx)
    return labels_with_idx

def print_count_per_class(dataset, idxs_per_user, type):
    idx_to_class = {val: key for key, val in dataset.class_to_idx.items()}
    set_sizes = {u: len(idxs_per_user[u]) for u in range(len(idxs_per_user))}
    print(f'The {type} set size is {set_sizes}')
    for u in range(len(idxs_per_user)):
        labels, counters = torch.unique(dataset.targets[idxs_per_user[u]],return_counts=True)
        print(f'[PAIV {u}]: {type} data distrib: {dict(zip([idx_to_class[x] for x in labels.tolist()], counters.tolist()))}')
    # count = { k: len(v) for k, v in idxs_per_user.items() }
    # print(count)

def dataset_iid_pablo(args,  num_users, force_balance=True):
    """
    Sample I.I.D. client data from MNIST dataset fixing the da
    :param args:
    :param num_users:
    :return: dict of image index
    """
    dataset,testset = get_dataset(args)

    num_items = args.fixed_size # number of items per users
    if num_items == -1:
        num_items = int(len(dataset)/num_users) # we use the whole dataset here

    train_idx_users = {node: [] for node in range(num_users)} # initialize dict with one empty list per user
    val_idx_users = {node: [] for node in range(num_users)} # initialize dict with one empty list per user


    if force_balance:
        # with force_balance set, we force a fixed number (num_items//num_labels) of images per class
        labels_with_idx = get_idxs_per_class(dataset)
        if num_items % len(labels_with_idx.keys()) > 0:
            print(f'*** The configured local dataset size is not a multiple of the number of classes, downsizing to a multiple\n')

        # get how many images you need to sample per label (note: this is integer division, so we discard the "remainder" size)
        size_per_label = num_items // len(labels_with_idx.keys())
        to_draw = size_per_label * num_users

        for c, idxs_list in labels_with_idx.items():
            # print(f'There are {len(idxs_list)} images for class {c}')
            to_draw = min(to_draw, len(idxs_list))
        # print(to_draw)

        size_per_label = to_draw // num_users
        size_per_label_train = int(np.floor(size_per_label * (1-args.val_split)))
        size_per_label_val = size_per_label - size_per_label_train

        if size_per_label_val < 1:
            print(f'The size of the validation set is ZERO')
            exit(1)
        
        # for each class...
        for c, idxs_list in labels_with_idx.items():
            # sample size_per_label * num_users of indices of images of that class
            # print(len(idxs_list))
            tmp = set(np.random.choice(idxs_list, to_draw, replace=False))
            # then assign size_per_label to each user
            for i in range(num_users):
                train_idx_users[i] += list(tmp)[i*size_per_label:i*size_per_label+size_per_label_train]
                val_idx_users[i] += list(tmp)[i*size_per_label+size_per_label_train:(i+1)*size_per_label]
        # # check the size
        # count = { k: len(v) for k, v in dict_users.items() }
        # print(count)
        # convert to tensor
        # train_idx_users = {k: torch.tensor(train_idx_users[k]) for k, v in train_idx_users.items()}
        # val_idx_users = {k: torch.tensor(val_idx_users[k]) for k, v in val_idx_users.items()}

        data_partitions = {u : {'train': torch.tensor(train_idx_users[u]), 'validation' : torch.tensor(val_idx_users[u])} for u in range(num_users)}
    else:
        # here is a true statistical iid: same sampling prob per class, but with small dataset the local samples generally differ slightly
        data_partitions = dict()
        all_idxs = [i for i in range(len(dataset))]
        train_size = int(np.floor(num_items * (1-args.val_split)))
        for i in range(num_users):
            # for each user, we draw uniformly at random num_items images, then split them into train and validation
            tmp = list(np.random.choice(all_idxs, num_items, replace=False))
            train_idx_users[i] = torch.tensor(tmp[:train_size])
            val_idx_users[i] = torch.tensor(tmp[train_size:])
            data_partitions[i] = {'train': train_idx_users[i], 'validation' : val_idx_users[i]}
            all_idxs = list(set(all_idxs) - set(tmp))

    print_count_per_class(dataset, train_idx_users, 'train')
    print_count_per_class(dataset, val_idx_users, 'validation')

    return dataset,testset, data_partitions

def mnist_noniid(dataset, num_users):
    """
    DEPRECATED
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 12000, 5
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    p = np.random.zipf(2.15, num_users)
    prob = p/np.sum(p)
    # divide and assign
    user_sequence = np.random.choice(
        range(num_users), num_shards, p=prob, replace=True)
    while len(set(idx_shard)):
        for i in user_sequence:
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    for uid, idxs in dict_users.items():
        print('id={}, len={}, labels={}'.format(
            uid, len(idxs), np.unique(np.array(labels[idxs]), return_counts=True)))
    return dict_users

def dataset_community_based(args, g):
    """
    Sample I.I.D. client data from MNIST dataset fixing the da
    :param args:
    :param the_graph:
    :return: TBD
    """
    dataset,testset = get_dataset(args)

    num_users = g.sai_graph.number_of_nodes()
    nodes_with_communities = g.get_nodes_with_communities()
    num_communities = 2 # compute

    num_items = args.fixed_size # number of items per users
    if num_items == -1:
        num_items = int(len(dataset)/num_users) # we use the whole dataset here

    train_idx_users = {node: [] for node in range(num_users)} # initialize dict with one empty list per user
    val_idx_users = {node: [] for node in range(num_users)} # initialize dict with one empty list per user


    if force_balance:
        # with force_balance set, we force a fixed number (num_items//num_labels) of images per class
        labels_with_idx = get_idxs_per_class(dataset)
        if num_items % len(labels_with_idx.keys()) > 0:
            print(f'*** The configured local dataset size is not a multiple of the number of classes, downsizing to a multiple\n')

        # get how many images you need to sample per label (note: this is integer division, so we discard the "remainder" size)
        size_per_label = num_items // len(labels_with_idx.keys())
        to_draw = size_per_label * num_users

        for c, idxs_list in labels_with_idx.items():
            # print(f'There are {len(idxs_list)} images for class {c}')
            to_draw = min(to_draw, len(idxs_list))
        # print(to_draw)

        size_per_label = to_draw // num_users
        size_per_label_train = int(np.floor(size_per_label * (1-args.val_split)))
        size_per_label_val = size_per_label - size_per_label_train

        if size_per_label_val < 1:
            print(f'The size of the validation set is ZERO')
            exit(1)
        
        # for each class...
        for c, idxs_list in labels_with_idx.items():
            # sample size_per_label * num_users of indices of images of that class
            # print(len(idxs_list))
            tmp = set(np.random.choice(idxs_list, to_draw, replace=False))
            # then assign size_per_label to each user
            for i in range(num_users):
                train_idx_users[i] += list(tmp)[i*size_per_label:i*size_per_label+size_per_label_train]
                val_idx_users[i] += list(tmp)[i*size_per_label+size_per_label_train:(i+1)*size_per_label]
        # # check the size
        # count = { k: len(v) for k, v in dict_users.items() }
        # print(count)
        # convert to tensor
        # train_idx_users = {k: torch.tensor(train_idx_users[k]) for k, v in train_idx_users.items()}
        # val_idx_users = {k: torch.tensor(val_idx_users[k]) for k, v in val_idx_users.items()}

        data_partitions = {u : {'train': torch.tensor(train_idx_users[u]), 'validation' : torch.tensor(val_idx_users[u])} for u in range(num_users)}
    else:
        # here is a true statistical iid: same sampling prob per class, but with small dataset the local samples generally differ slightly
        data_partitions = dict()
        all_idxs = [i for i in range(len(dataset))]
        train_size = int(np.floor(num_items * (1-args.val_split)))
        for i in range(num_users):
            # for each user, we draw uniformly at random num_items images, then split them into train and validation
            tmp = list(np.random.choice(all_idxs, num_items, replace=False))
            train_idx_users[i] = torch.tensor(tmp[:train_size])
            val_idx_users[i] = torch.tensor(tmp[train_size:])
            data_partitions[i] = {'train': train_idx_users[i], 'validation' : val_idx_users[i]}
            all_idxs = list(set(all_idxs) - set(tmp))

    print_count_per_class(dataset, train_idx_users, 'train')
    print_count_per_class(dataset, val_idx_users, 'validation')

    return dataset,testset, data_partitions

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(
            all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def dirichlet_dataset_partitioner(args, num_users=10,dirichlet_alpha=None, seed=42, verbose=True):
    print(f'Dirichlet partioning')
    if args.dataset == 'mnist':
        trainset = datasets.MNIST('../data/mnist/', train=True, download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize(
                                               (0.1307,), (0.3081,))
                                       ]))
        testset = datasets.MNIST('../data/mnist/', train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          (0.1307,), (0.3081,))
                                  ]))

        label_partition = MNISTPartitioner(trainset.targets,
                                        num_clients=num_users,
                                        partition="noniid-labeldir",
                                        dir_alpha=dirichlet_alpha,
                                        seed=seed,
                                        verbose=verbose
                                        )
    elif args.dataset == 'fashion_mnist':

        trainset = datasets.FashionMNIST(
            '../data/fashion/', train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,), (0.3081,))
            ]))
        testset = datasets.FashionMNIST(
            '../data/fashion/', train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,), (0.3081,))
            ]))

        label_partition = FMNISTPartitioner(trainset.targets,
                                           num_clients=num_users,
                                           partition="noniid-labeldir",
                                           dir_alpha=dirichlet_alpha,
                                           seed=seed,
                                           verbose=verbose
                                           )
    elif args.dataset == 'cifar10':
        trainset = datasets.CIFAR10(
            '../data/cifar10/', train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor()]))
        testset = datasets.CIFAR10(
            '../data/cifar10/', train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor()]))
        label_partition = CIFAR10Partitioner(trainset.targets,
                                            num_clients=num_users,
                                            balance=None,
                                            partition="dirichlet",
                                            dir_alpha=dirichlet_alpha,
                                            seed=seed,
                                            verbose=verbose
                                            )

    elif args.dataset == 'cifar100':
        trainset = datasets.CIFAR100(
            '../data/cifar100/', train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor()]))
        testset = datasets.CIFAR100(
            '../data/cifar100/', train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor()]))
        label_partition = CIFAR100Partitioner(trainset.targets,
                                             num_clients=num_users,
                                             balance=None,
                                             partition="dirichlet",
                                             dir_alpha=dirichlet_alpha,
                                             seed=seed,
                                             verbose=verbose
                                             )
    
    # Split training data in train and validation sets
    partitions = label_partition.client_dict
    new_data_partitions =dict()
    # initialize dict with one empty list per user
    train_idx_users = {node: [] for node in range(num_users)}
    # initialize dict with one empty list per user
    val_idx_users = {node: [] for node in range(num_users)}

    for i, idxs in partitions.items():
        # print(f'Node {i} has {len(idxs)} indices assigned')
        train_size = int(np.floor(len(idxs)) * (1-args.val_split))
        randidx = torch.randperm(len(idxs))
        # print(randidx.size())
        train_idx_users[i] = torch.tensor(idxs[randidx[:train_size]])
        val_idx_users[i] = torch.tensor(idxs[randidx[train_size:]])
        new_data_partitions[i] = {
            'train': train_idx_users[i], 'validation': val_idx_users[i]}

    #################################################################################################################

    if not isinstance(trainset.targets, torch.Tensor):
        trainset.targets = torch.tensor(trainset.targets)


    print_count_per_class(trainset, train_idx_users, 'train')
    print_count_per_class(trainset, val_idx_users, 'validation')
    


    return trainset,testset,new_data_partitions






if __name__ == '__main__':
    # dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
    #                                transform=transforms.Compose([
    #                                    transforms.ToTensor(),
    #                                    transforms.Normalize(
    #                                        (0.1307,), (0.3081,))
    #                                ]))
    # num = 100
    # d = mnist_noniid(dataset_train, num)
    num_clients = 10
    args = args_parser()
    trainset,testset,users_dict = dirichlet_dataset_partitioner(args,num_users=num_clients,dirichlet_alpha=1e4,seed=42)
    print(partition_report(trainset.targets,users_dict))
    


