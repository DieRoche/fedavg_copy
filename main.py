from collections import OrderedDict, Counter, defaultdict

import numpy as np
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict,  Any


from Codification import compress_parameters, decompress_parameters
from Config import (train_dataset, dev_dataset,num_clients, USE_pathological, COMPRESS_method, USE_COMPRESSION, num_classes_per_client,
                    classes, client_fraction, rounds, batch_size, epochs_per_client, learning_rate,
                    total_downlink, total_uplink_compressed , total_uplink_uncompressed,
                    total_train_size, total_test_size, total_eval_size, generate_report_filename)
from Report import initialize_csv, log_round
from DataReqs import split_data

import matplotlib.pyplot as plt
from collections import Counter
import os
import copy

from fedpruning import prune_by_percentile_hybrid, init_masks, mask_model_hybrid

def get_class_distribution(dataset):
    counts = defaultdict(int)
    for _, label in dataset:
        counts[label] += 1
    return dict(sorted(counts.items()))


examples_per_client = total_train_size // num_clients
client_datasets = split_data(
    train_dataset,
    pathological=USE_pathological,
    num_clients=num_clients,
    num_classes_per_client=num_classes_per_client
)

report_name = generate_report_filename()
report_path = "./" + report_name
initialize_csv(report_path, use_compression= USE_COMPRESSION )

print("--------------------------------------------------------------------------------------------")
print(f"Total train {total_train_size}, Total eval {total_eval_size}, Total test {total_test_size}")

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader(DataLoader):
        def __init__(self, dl, device):
            self.dl = dl
            self.device = device

        def __iter__(self):
            for batch in self.dl:
                yield to_device(batch, self.device)

        def __len__(self):
            return len(self.dl)

device = get_device()

class FederatedNet(torch.nn.Module):    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.track_layers = {'conv1': self.conv1, 'conv2': self.conv2, 'fc1': self.fc1, 'fc2': self.fc2}
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
    
    def get_track_layers(self):
        return self.track_layers
    
    def apply_parameters(self, parameters_dict: Dict[str, Any]):
    #"""Apply parameters to model layers, ensuring device consistency."""
        with torch.no_grad():
            for layer_name in parameters_dict:
                layer = self.track_layers[layer_name]
                
                # Ensure the parameters are moved to the layer's device
                weight = parameters_dict[layer_name]['weight'].to(layer.weight.device)
                bias = parameters_dict[layer_name]['bias'].to(layer.bias.device)
                
                # Apply the parameters
                layer.weight.data.zero_()
                layer.bias.data.zero_()
                layer.weight.data += weight
                layer.bias.data += bias
    
    def get_parameters(self):
        parameters_dict = dict()
        for layer_name in self.track_layers:
            parameters_dict[layer_name] = {
                'weight': self.track_layers[layer_name].weight.data, 
                'bias': self.track_layers[layer_name].bias.data
            }
        return parameters_dict
    
    def batch_accuracy(self, outputs, labels):
        with torch.no_grad():
            _, predictions = torch.max(outputs, dim=1)
            return torch.tensor(torch.sum(predictions == labels).item() / len(predictions))
    
    def _process_batch(self, batch):
        images, labels = batch
        outputs = self(images)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        accuracy = self.batch_accuracy(outputs, labels)
        return (loss, accuracy)
    
    ##################
    ### Fedpruning ###
    # Included freeze, pruning mask 
    ##################

    def fit(self, dataset, epochs, lr, batch_size=128, opt=torch.optim.SGD, freeze= False, pruning_mask=None):
        dataloader = DeviceDataLoader(DataLoader(dataset, batch_size, shuffle=True), device)
        optimizer = opt(self.parameters(), lr)
        history = []
        
        for epoch in range(epochs):
            losses = []
            accs = []
            for batch in dataloader:
                loss, acc = self._process_batch(batch)
                loss.backward()
                
                 ###### Fedpruning ######
                if freeze and pruning_mask is not None:
                    for name, param in self.named_parameters():
                        # Only process weight parameters
                        if 'weight' not in name:
                            continue
                            
                        # Get layer name from parameter name
                        layer_name = name.replace('.weight', '')
                        
                        if layer_name in pruning_mask:
                            mask = pruning_mask[layer_name]['weight']
                            
                            # Convert to tensor if needed
                            if isinstance(mask, np.ndarray):
                                mask = torch.from_numpy(mask).to(param.device)
                            else:
                                mask = mask.to(param.device)
                            
                            # Handle Conv2d vs Linear differently
                            if 'conv' in layer_name:
                                # For Conv2d: reshape mask to [out_channels, 1, 1, 1] for broadcasting
                                mask = mask.view(-1, 1, 1, 1)
                            # Linear layers don't need reshaping
                            
                            if param.grad is not None:
                                param.grad.data *= mask
                ########################

                
                optimizer.step()
                optimizer.zero_grad()
                loss.detach()
                losses.append(loss)
                accs.append(acc)
            avg_loss = torch.stack(losses).mean().item()
            avg_acc = torch.stack(accs).mean().item()
            history.append((avg_loss, avg_acc))

        ####################
        ### Fedpruning ###
        ###################
        if freeze and pruning_mask is not None:
            prune_by_percentile_hybrid(self, pruning_mask, percent=20)
        ##################


        return history
    
    def evaluate(self, dataset, batch_size=128):
        dataloader = DeviceDataLoader(DataLoader(dataset, batch_size), device)
        losses = []
        accs = []
        with torch.no_grad():
            for batch in dataloader:
                loss, acc = self._process_batch(batch)
                losses.append(loss)
                accs.append(acc)
        avg_loss = torch.stack(losses).mean().item()
        avg_acc = torch.stack(accs).mean().item()
        return (avg_loss, avg_acc)

    def compute_density(self):
        total_params = 0
        nonzero_params = 0
        for name, param in self.named_parameters():
            if 'weight' in name and ('conv' in name or 'fc' in name):
                total_params += param.numel()
                nonzero_params += torch.count_nonzero(param).item()
        return nonzero_params / total_params


class Client:
    def __init__(self, client_id, dataset, mask=None,compress_method = COMPRESS_method):
        self.client_id = client_id
        self.dataset = dataset
        self.compress_method = compress_method
        #For fedpruning method
        self.mask = mask
    
    def get_dataset_size(self):
        return len(self.dataset)
    
    def get_client_id(self):
        return self.client_id
    
    def train(self, parameters_dict):
        net = to_device(FederatedNet(), device)
        net.apply_parameters(parameters_dict)

        #########################
        ### Fedpruning ###
        # aded freeze and pruning mask
        if self.mask is not None:
            mask_model_hybrid(net, self.mask, initial_state_dict)

        train_history = net.fit(self.dataset, epochs_per_client, learning_rate, batch_size,freeze= True, pruning_mask=self.mask)
        print('{}: Loss = {}, Accuracy = {}'.format(self.client_id, round(train_history[-1][0], 4), round(train_history[-1][1], 4)))
        new_params = net.get_parameters()
        
        if USE_COMPRESSION:
            compressed_update = compress_parameters(new_params,method=self.compress_method)
            del new_params
            torch.cuda.empty_cache()
            return compressed_update

        return {'params': new_params,
                'mask': self.mask,
                'density': net.compute_density()}


#print("\nDetailed Client Dataset Information:")
for i, client_dataset in enumerate(client_datasets):
    # Get all labels in this client's dataset
    labels = [label for _, label in client_dataset]
    class_dist = get_class_distribution(client_dataset)
    
    #print(f"\nClient {i} ({len(client_dataset)} total images):")
    for digit in range(10):
        count = class_dist.get(digit, 0)
        #print(f"  Digit {digit}: {count} images ({count/len(client_dataset):.1%})")

def get_parameters_size(parameters_dict):
    total_bytes = 0 
    for layer_name in parameters_dict:
        weight_size = parameters_dict[layer_name]['weight'].nelement() * parameters_dict[layer_name]['weight'].element_size()
        bias_size = parameters_dict[layer_name]['bias'].nelement() * parameters_dict[layer_name]['bias'].element_size()
        total_bytes += weight_size + bias_size
    return total_bytes  

def get_sparse_size(parameters_dict, masks):
    total_bytes = 0
    for layer_name in parameters_dict:
        # Only count non-zero elements
        weight_mask = masks[layer_name]['weight']
        bias_mask = masks[layer_name].get('bias', None)
        
        weight_nonzero = parameters_dict[layer_name]['weight'][weight_mask > 0]
        total_bytes += weight_nonzero.numel() * weight_nonzero.element_size()
        
        if bias_mask is not None:
            bias_nonzero = parameters_dict[layer_name]['bias'][bias_mask > 0]
            total_bytes += bias_nonzero.numel() * bias_nonzero.element_size()
    
    return total_bytes

def sparse_aggregate(updates):
    global_params = defaultdict(lambda: {'weight': 0, 'bias': 0})
    mask_counts = defaultdict(lambda: {'weight': 0, 'bias': 0})
    
    for update in updates:
        for layer in update['params']:
            # Get weight and mask
            weight = update['params'][layer]['weight']
            weight_mask = update['mask'][layer]['weight']
            
            # Handle Conv2d vs Linear differently
            if 'conv' in layer:  # Conv2d layer
                # Reshape mask for proper broadcasting [out_c, 1, 1, 1]
                reshaped_mask = weight_mask.view(-1, 1, 1, 1)
                global_params[layer]['weight'] += weight * reshaped_mask
                mask_counts[layer]['weight'] += reshaped_mask
            else:  # Linear layer
                global_params[layer]['weight'] += weight * weight_mask
                mask_counts[layer]['weight'] += weight_mask
            
            # Handle bias if exists
            if 'bias' in update['params'][layer]:
                bias = update['params'][layer]['bias']
                bias_mask = update['mask'][layer]['bias']
                global_params[layer]['bias'] += bias * bias_mask
                mask_counts[layer]['bias'] += bias_mask
    
    # Normalize
    for layer in global_params:
        # Normalize weights
        global_params[layer]['weight'] = torch.where(
            mask_counts[layer]['weight'] > 0,
            global_params[layer]['weight'] / mask_counts[layer]['weight'],
            torch.zeros_like(global_params[layer]['weight'])
        )
        
        # Normalize biases
        if 'bias' in global_params[layer]:
            global_params[layer]['bias'] = torch.where(
                mask_counts[layer]['bias'] > 0,
                global_params[layer]['bias'] / mask_counts[layer]['bias'],
                torch.zeros_like(global_params[layer]['bias'])
            )
    
    return dict(global_params)

clients = [Client('client_' + str(i), client_datasets[i]) for i in range(num_clients)]

global_net = to_device(FederatedNet(), device)

initial_state_dict = global_net.state_dict()
for client in clients:
    client.mask = init_masks(global_net)  # <-- Add this!
    client.init_state_dict = copy.deepcopy(initial_state_dict)
    temp_model = to_device(FederatedNet(), device)
    mask_model_hybrid(temp_model, client.mask, client.init_state_dict) 

for i in range(rounds):
    print('\n ------------------------------------------')
    print('Start Round {} ...'.format(i + 1))
    curr_parameters = global_net.get_parameters()
    parameters_size = get_parameters_size(curr_parameters)
    num_participating = max(1, int(client_fraction * num_clients))
    participating_clients = random.sample(clients, num_participating)
    
    round_downlink = parameters_size * num_clients  # All clients receive model
    total_downlink += round_downlink

    new_parameters = dict([(layer_name, {'weight': 0, 'bias': 0}) for layer_name in curr_parameters])
    round_uplink_compressed = 0
    round_uplink_uncompressed = 0

    weight_bytes = b''.join(
        layer['weight'].cpu().numpy().tobytes() 
        for layer in curr_parameters.values()
    )

    byte_counts = Counter(weight_bytes)
    total_bytes = len(weight_bytes)
    byte_frequencies = [count / total_bytes for count in byte_counts.values()]
      
    byte_counts = Counter(weight_bytes)
    total_bytes = len(weight_bytes)
    entropy = -sum((count/total_bytes) * math.log2(count/total_bytes) 
                  for count in byte_counts.values())
    
    print(f"\nEntropy Analysis:")
    print(f"- Unique bytes: {len(byte_counts)}")
    print(f"- Entropy: {entropy:.2f}/8 bits per byte")
    print(f"- Theoretical max compression: {(8-entropy)/8*100:.2f}%")
    print(f"- Current size: {total_bytes/1024:.2f} KB")

    client_updates = []
    for client in participating_clients:
        update = client.train(curr_parameters)
        client_updates.append(update)

        if USE_COMPRESSION:
            client_params = decompress_parameters(update)
            round_uplink_compressed += len(update['compressed_data'])
            round_uplink_uncompressed += get_parameters_size(client_params)
        else:
            # Use raw parameters
            client_params = update['params']
            transmitted_size = get_parameters_size(client_params)
            round_uplink_compressed += transmitted_size
            round_uplink_uncompressed += transmitted_size


        fraction = client.get_dataset_size() / total_train_size

        for layer_name in client_params:
            new_parameters[layer_name]['weight'] += fraction * client_params[layer_name]['weight']
            new_parameters[layer_name]['bias'] += fraction * client_params[layer_name]['bias']
        del client_params
    total_uplink_compressed += round_uplink_compressed
    total_uplink_uncompressed += round_uplink_uncompressed

    new_parameters = sparse_aggregate(client_updates)

    global_net.apply_parameters(new_parameters)
    density = global_net.compute_density()
    print(f'Global model density after round {i + 1}: {density:.4f}')
    train_loss, train_acc = global_net.evaluate(train_dataset)
    dev_loss, dev_acc = global_net.evaluate(dev_dataset)
    print(f'After round {i + 1}:')
    print(f'  Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}')
    print(f'  Dev   - Loss: {dev_loss:.4f}, Accuracy: {dev_acc:.4f}\n')

    global_mask = average_mask([u['mask'] for u in client_updates])
    
    for client in clients:
        client.mask = global_mask
        
    metrics_dict = {
        "train_loss": train_loss,
        "dev_loss": dev_loss,
        "train_acc": train_acc,
        "dev_acc": dev_acc,
        "avg_uplink": round_uplink_compressed / num_participating,
        "avg_downlink": round_downlink / num_clients,
        "total_uplink": total_uplink_compressed,
        "total_downlink": total_downlink,
        "density": density,
    }

    print(f'\n[Communication]')
    print(f'Downlink (All {num_clients} clients): {round_downlink/1024:.2f} KB')
    print(f'Uplink ({num_participating} clients): {round_uplink_compressed/1024:.2f} KB')
    
    if USE_COMPRESSION:
        metrics_dict["uncompressed_size"] = round_uplink_uncompressed / 1024  # in KB
        metrics_dict["compression_ratio"] = (
        round_uplink_uncompressed / round_uplink_compressed
        if round_uplink_compressed > 0 else 0
            )
        print(f'  (Uncompressed equivalent: {round_uplink_uncompressed/1024:.2f} KB)')
        print(f'  Compression ratio: {round_uplink_uncompressed/round_uplink_compressed:.2f}x')
    print(f'Cumulative Downlink: {total_downlink/1024:.2f} KB')
    print(f'Cumulative Uplink: {total_uplink_compressed/1024:.2f} KB')
    
    log_round(report_path, i + 1, metrics_dict, use_compression=USE_COMPRESSION)


    if USE_COMPRESSION:
        print(f'Total bandwidth saved: {(total_uplink_uncompressed - total_uplink_compressed)/1024:.2f} KB')

    del new_parameters, curr_parameters
    
    
