import os
import time
import argparse
import torch
import json
import copy
import numpy as np
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import random
import model as mdl

import torch.distributed as dist

device = "cpu"
torch.set_num_threads(4)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

batch_size = 64  # batch for one node
def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """

    running_loss = 0.0
    average_time_per_iteration = []
    for batch_idx, (data, target) in enumerate(train_loader):
        # print(data[0], target[0])
        start = time.time()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        for param in model.parameters():
            gradient = param.grad
            gathered_gradient = [torch.zeros_like(gradient) for _ in range(args.num_nodes)]
            if args.rank == 0:
                dist.gather(gradient, gathered_gradient, dst=0)
            else:
                dist.gather(gradient, dst=0)

            avg_gradients = torch.zeros_like(gradient)
            if args.rank == 0:
                avg_gradients = torch.stack(gathered_gradient).mean(dim=0)
                dist.scatter(param.grad.data, [avg_gradients for _ in range(args.num_nodes)], src=0)
            else:
                dist.scatter(param.grad.data, src=0)
        
        optimizer.step()

        stop = time.time()
        if batch_idx < 40:  # only store the first 40 iterations
            average_time_per_iteration.append(stop-start)

        running_loss += loss.item()
        if batch_idx % 20 == 0:  # print statistics after every 20 iterations
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.3f}'.format(
                epoch + 1, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), running_loss / 20))
            running_loss = 0.0
    
    # print the average time per iteration for the first 40 iterations, discarding the first
    print(f'Average time per iteration: {np.mean(average_time_per_iteration[1:]):.2f} +- '
           f'{np.std(average_time_per_iteration[1:]):.2f}')
    return None

def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


def main(rank):
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
    training_set = datasets.CIFAR10(root="./data", train=True,
                                                download=True, transform=transform_train)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(training_set,
                                                                    rank=rank,
                                                                    seed=0)
    train_loader = torch.utils.data.DataLoader(training_set,
                                                    num_workers=2,
                                                    batch_size=batch_size,
                                                    sampler=train_sampler,
                                                    shuffle=False,
                                                    pin_memory=True)
    
    test_set = datasets.CIFAR10(root="./data", train=False,
                                download=True, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              num_workers=2,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True)
    training_criterion = torch.nn.CrossEntropyLoss().to(device)

    model = mdl.VGG11()
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=0.0001)
    
    # running training for one epoch
    for epoch in range(1):
        train_model(model, train_loader, optimizer, training_criterion, epoch)
        test_model(model, test_loader, training_criterion)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--master-ip', type=str, default='172.18.0.2:6585') 
    parser.add_argument('-r', '--rank', type=int)
    parser.add_argument('-n', '--num-nodes', type=int, default=4)

    args = parser.parse_args()

    dist.init_process_group('gloo', init_method='tcp://{}'.format(args.master_ip),
                        world_size=args.num_nodes, rank=args.rank)
    main(args.rank)

