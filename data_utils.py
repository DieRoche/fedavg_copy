import json
import itertools
import re
import string
import unicodedata
import random

import torch
import numpy as np
from torch.utils.data import Subset
from torchvision import datasets, transforms
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

nlp_datasets = ["agnews", "imdb", "sogou"]

import sys
import csv
maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)


def split_noniid(train_idcs, train_labels, alpha, n_clients):
    '''
    Splits a list of data indices with corresponding labels
    into subsets according to a dirichlet distribution with parameter
    alpha
    '''
    n_classes = train_labels.max() + 1
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)

    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]

    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [train_idcs[np.concatenate(idcs)] for idcs in client_idcs]

    return client_idcs, label_distribution


class CustomSubset(Subset):
    '''A custom subset class with customizable data transformation'''

    def __init__(self, dataset, indices, subset_transform=None):
        super().__init__(dataset, indices)
        self.subset_transform = subset_transform

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]

        if self.subset_transform is not None:
            x = self.subset_transform(x)

        return x, y

def plot_client_distributions(client_distribution, label_distribution, n_classes, save_path='client_distributions.png'):
    '''
    Plot bar charts showing the label distribution for each client
    '''
    n_clients = len(client_distribution)
    
    # Create subplots
    fig, axes = plt.subplots(2, (n_clients + 1) // 2, figsize=(15, 8))
    if n_clients == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    class_names = [f'Class {i}' for i in range(n_classes)]
    
    for i in range(n_clients):
        ax = axes[i]
        bars = ax.bar(range(n_classes), client_distribution[i], alpha=0.7)
        ax.set_title(f'Client {i}')
        ax.set_xlabel('Classes')
        ax.set_ylabel('Proportion')
        ax.set_xticks(range(n_classes))
        ax.set_xticklabels(class_names, rotation=45)
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0.01:  # Only show labels for non-negligible values
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Hide empty subplots
    for i in range(n_clients, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to prevent display
    print(f"Client distributions plot saved to: {save_path}")

def inspect_client_data(client_data, client_idx=0, n_samples=10, dataset_name='mnist', save_path='client_inspection.png'):
    '''
    Inspect a specific client's data by showing sample images and their labels
    '''
    client_subset = client_data[client_idx]
    
    # Get sample indices
    sample_indices = np.random.choice(len(client_subset), min(n_samples, len(client_subset)), replace=False)
    
    # Create subplot for images
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.flatten()
    
    labels_found = []
    
    for i, idx in enumerate(sample_indices):
        if i >= 10:  # Limit to 10 samples
            break
            
        image, label = client_subset[idx]
        labels_found.append(label)
        
        # Convert tensor to numpy if needed
        if hasattr(image, 'numpy'):
            img_np = image.numpy()
        else:
            img_np = np.array(image)
        
        # Handle different image formats
        if dataset_name.lower() == 'mnist':
            if len(img_np.shape) == 3:
                img_np = img_np.squeeze()  # Remove channel dimension
            axes[i].imshow(img_np, cmap='gray')
        elif dataset_name.lower() == 'cifar10':
            if len(img_np.shape) == 3 and img_np.shape[0] == 3:
                img_np = np.transpose(img_np, (1, 2, 0))  # CHW to HWC
            # Denormalize CIFAR10 if normalized
            if img_np.min() < 0:  # Likely normalized
                mean = np.array([0.4914, 0.4822, 0.4465])
                std = np.array([0.2023, 0.1994, 0.2010])
                img_np = img_np * std + mean
                img_np = np.clip(img_np, 0, 1)
            axes[i].imshow(img_np)
        
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(sample_indices), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Client {client_idx} - Sample Data (Total samples: {len(client_subset)})')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to prevent display
    print(f"Client {client_idx} inspection plot saved to: {save_path}")
    
    # Print label distribution for this client
    unique_labels, counts = np.unique(labels_found, return_counts=True)
    print(f"\nSample label distribution for Client {client_idx}:")
    for label, count in zip(unique_labels, counts):
        print(f"  Label {label}: {count} samples")

def get_dataset(args):
    if args.dataset == 'mnist':
        transform = transforms.Compose([transforms.ToTensor()])
        data = datasets.MNIST(root=".", download=True, transform=transform)
        n_classes = 10
    elif args.dataset == 'cifar10':
        n_classes = 10
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        data = datasets.CIFAR10(root=".", download=True, transform=transform)

    elif args.dataset == 'cifar100':
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408),
                    (0.2675, 0.2565, 0.2761)
                ),
            ]
        )
        data = datasets.CIFAR100(root=".", download=True, transform=transform)
        n_classes = 100
    else:
        raise NotImplementedError

    idcs = np.random.permutation(len(data))
    if args.dataset == 'emnist':
        train_idcs, test_idcs = idcs[:50000], idcs[50000:60000]
    else:
        split = int(len(data) * args.train_frac)
        train_idcs, test_idcs = idcs[:split], idcs[split:]

    if args.dataset in nlp_datasets:
        train_labels, formatted_data = [], []
        for l, t in data:
            token = vocab(tokenizer(t))
            label = l - 1
            formatted_data.append((token, label))
            train_labels.append(label)
        train_labels = np.array(train_labels)
        data = formatted_data
    else:
        if hasattr(data, 'targets'):
            all_labels = np.array(data.targets)
            train_labels = all_labels[train_idcs]
        elif hasattr(data, 'labels'):
            all_labels = np.array(data.labels)
            train_labels = all_labels[train_idcs]
        else:
            train_labels=[]
            for idx in train_idcs:
                _, label = data[idx]
                train_labels.append(label)
            train_labels = np.array(train_labels)
    
    client_idcs, label_distribution = split_noniid(train_idcs, train_labels, alpha=args.dirichlet, n_clients=args.n_client)

    split_labels = [{} for _ in range(args.n_client)]
    client_distribution = []
    for i, idcs in enumerate(client_idcs):
        for idx in idcs:
            if hasattr(data, 'targets'):
                label = data.targets[idx]
            elif hasattr(data, 'labels'):
                label = data.labels[idx]
            else:
                _, label = data[idx]
            split_labels[i][label] = split_labels[i].get(label, 0) + 1

        total_samples = len(idcs)
        client_dist = np.zeros(n_classes)
        for label, count in split_labels[i].items():
            client_dist[label] = count / total_samples
        client_distribution.append(client_dist)
        print("Client %d: %s" % (i, split_labels[i]))

    client_train_data = []
    client_val_data = []
    val_split = 0.2

    for idcs in client_idcs:
        shuffled_indices = np.random.permutation(idcs)
        split_point = int(len(shuffled_indices) * (1 - val_split))

        if len(shuffled_indices) > 0 and split_point == 0:
            split_point = 1
        if split_point >= len(shuffled_indices):
            split_point = max(len(shuffled_indices) - 1, 0)

        train_indices = shuffled_indices[:split_point] if split_point > 0 else np.array([], dtype=int)
        val_indices = shuffled_indices[split_point:]

        client_train_data.append(CustomSubset(data, train_indices))
        client_val_data.append(CustomSubset(data, val_indices))

    test_data = CustomSubset(data, test_idcs)

    plot_client_distributions(client_distribution, label_distribution, n_classes, save_path='client_distributions.png')

    if len(client_train_data) > 0:
        inspect_client_data(client_train_data, client_idx=0, n_samples=10, dataset_name=args.dataset, save_path='client_0_inspection.png')

    return client_train_data, client_val_data, test_data, n_classes, client_distribution, label_distribution

