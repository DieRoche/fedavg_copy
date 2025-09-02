import copy
import random
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import get_config
from data_utils import get_dataset
from resnet18 import ResNet18


def client_update(model, loader, epochs, device, lr):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(epochs):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
    return model.state_dict()


def evaluate(model, loader, device):
    model.eval()
    loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return loss / total, correct / total


def fedavg(state_dicts, weights):
    avg = OrderedDict()
    for k in state_dicts[0].keys():
        avg[k] = sum(weight * sd[k] for sd, weight in zip(state_dicts, weights))
    return avg


def main():
    args = get_config()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    client_data, test_data, n_classes, _, _ = get_dataset(args)
    global_model = ResNet18(num_classes=n_classes).to(device)

    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    n_clients = len(client_data)

    for round_idx in range(args.n_epoch):
        m = max(1, int(args.client_fraction * n_clients))
        selected = random.sample(range(n_clients), m)

        local_states = []
        local_sizes = []

        for idx in selected:
            local_model = copy.deepcopy(global_model)
            loader = DataLoader(client_data[idx], batch_size=args.batch_size, shuffle=True)
            state_dict = client_update(local_model, loader, args.n_client_epoch, device, args.lr)
            local_states.append(state_dict)
            local_sizes.append(len(client_data[idx]))

        weights = [size / sum(local_sizes) for size in local_sizes]
        global_state = fedavg(local_states, weights)
        global_model.load_state_dict(global_state)

        loss, acc = evaluate(global_model, test_loader, device)
        print(f"Round {round_idx + 1:03d}: loss={loss:.4f}, accuracy={acc:.4f}")

    print("Training complete.")


if __name__ == "__main__":
    main()
