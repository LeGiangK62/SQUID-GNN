import torch
import os
from torch_geometric.datasets import TUDataset, Planetoid
from torch_geometric.loader import DataLoader

def load_dataset(name, path='../data', train_size=None, test_size=None, batch_size=32):
    name = name.upper()
    
    if name in ['MUTAG', 'ENZYMES', 'PROTEINS']:
        dataset = TUDataset(os.path.join(path, 'TUDataset'), name=name)
        torch.manual_seed(1712)
        dataset = dataset.shuffle()
        if train_size and test_size:
            train_dataset = dataset[:train_size]
            test_dataset = dataset[train_size:train_size + test_size]
        else:
            train_dataset = dataset[:int(0.8 * len(dataset))]
            test_dataset = dataset[int(0.8 * len(dataset)):]
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        task_type = 'graph'
    elif name in ['CORA', 'CITESEER', 'PUBMED']:
        dataset = Planetoid(root=os.path.join(path, 'Planetoid'), name=name)
        data = dataset[0]  # only one graph
        train_loader = test_loader = data  # use the same for node-level
        task_type = 'node'
    else:
        raise ValueError(f"Dataset '{name}' not supported.")
    
    return dataset, train_loader, test_loader, task_type

def eval_dataset(name, path='../data', eval_size=None, batch_size=32, seed=1309):
    name = name.upper()
    
    if name in ['MUTAG', 'ENZYMES', 'PROTEINS']:
        dataset = TUDataset(os.path.join(path, 'TUDataset'), name=name)
        torch.manual_seed(seed)
        dataset = dataset.shuffle()
        if eval_size:
            eval_set = dataset[:eval_size]
        else:
            eval_set = dataset[int(0.8 * len(dataset)):]
        eval_loader = DataLoader(eval_set, batch_size=batch_size)
        task_type = 'graph'
    elif name in ['CORA', 'CITESEER', 'PUBMED']:
        dataset = Planetoid(root=os.path.join(path, 'Planetoid'), name=name)
        data = dataset[0]  # only one graph
        eval_loader = data  # use the same for node-level
        task_type = 'node'
    else:
        raise ValueError(f"Dataset '{name}' not supported.")
    
    return eval_loader


# def load_dataset(name, path='../data', train_size=None, eval_size=None, test_size=None, batch_size=32):
#     name = name.upper()

#     if name in ['MUTAG', 'ENZYMES', 'PROTEINS']:
#         dataset = TUDataset(os.path.join(path, 'TUDataset'), name=name)
#         torch.manual_seed(1712)
#         dataset = dataset.shuffle()

#         total_len = len(dataset)
#         if train_size and eval_size and test_size:
#             train_dataset = dataset[:train_size]
#             eval_dataset = dataset[train_size:train_size + eval_size]
#             test_dataset = dataset[train_size + eval_size:train_size + eval_size + test_size]
#         else:
#             train_end = int(0.7 * total_len)
#             eval_end = int(0.85 * total_len)
#             train_dataset = dataset[:train_end]
#             eval_dataset = dataset[train_end:eval_end]
#             test_dataset = dataset[eval_end:]

#         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#         eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
#         test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#         task_type = 'graph'

#     elif name in ['CORA', 'CITESEER', 'PUBMED']:
#         dataset = Planetoid(root=os.path.join(path, 'Planetoid'), name=name)
#         data = dataset[0]  # only one graph
#         train_loader = eval_loader = test_loader = data
#         task_type = 'node'
#     else:
#         raise ValueError(f"Dataset '{name}' not supported.")

#     return dataset, train_loader, eval_loader, test_loader, task_type

