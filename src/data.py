import torch
import os
from torch_geometric.datasets import TUDataset, ZINC, Planetoid, WikipediaNetwork, IMDB, DBLP, AMiner, Flickr, WebKB
from torch_geometric.loader import DataLoader
from utils import FixZINC, ConstantX, DegreeX, WebKBPreprocess

def load_dataset(name, path='../data', train_size=None, test_size=None, eval_size=None, batch_size=32):
    name = name.upper()
    task_type = 'graph'

    if name in ['MUTAG', 'ENZYMES', 'PROTEINS']:
        dataset = TUDataset(os.path.join(path, 'TUDataset'), name=name)
        torch.manual_seed(1712)
        dataset = dataset.shuffle()
    elif name == 'REDDIT-BINARY':
        dataset = TUDataset(os.path.join(path, 'TUDataset'), name=name, transform=ConstantX())
        torch.manual_seed(1712)
        dataset = dataset.shuffle()

    elif name == 'ZINC':
        dataset = ZINC(root=os.path.join(path, 'ZINC'), transform=FixZINC())
        torch.manual_seed(1712)
        dataset = dataset.shuffle()

    elif name in ['CORA', 'CITESEER', 'PUBMED']:
        dataset = Planetoid(root=os.path.join(path, 'Planetoid'), name=name)
        data = dataset[0]
        
        torch.manual_seed(1712)
        num_nodes = data.num_nodes
        perm = torch.randperm(num_nodes)
        if train_size:
            train_idx = perm[:train_size]
            data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            data.train_mask[train_idx] = True
        if eval_size:
            val_idx = perm[train_size:train_size + eval_size]
            data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            data.val_mask[val_idx] = True
        if test_size:
            test_idx = perm[train_size + eval_size:train_size + eval_size + test_size]
            data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            data.test_mask[test_idx] = True
        return dataset, data, data, 'node'

    elif name in ['CORNELL', 'WISCONSIN']:
        dataset = WebKB(root=os.path.join(path, 'WebKB'), name=name.lower())
        data = dataset[0]
        torch.manual_seed(1712)
        num_nodes = data.num_nodes
        perm = torch.randperm(num_nodes)
        if train_size:
            train_idx = perm[:train_size]
            data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            data.train_mask[train_idx] = True
        if eval_size:
            val_idx = perm[train_size:train_size + eval_size]
            data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            data.val_mask[val_idx] = True
        if test_size:
            test_idx = perm[train_size + eval_size:train_size + eval_size + test_size]
            data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            data.test_mask[test_idx] = True
        return dataset, data, data, 'node'
    
    elif name in ['CHAMELEON', 'SQUIRREL', 'CROCODILE']:
        dataset = dataset = WikipediaNetwork(root=os.path.join(path, 'Wiki'),name=name.lower(), geom_gcn_preprocess=True if name != 'CROCODILE' else False) 
        data = dataset[0]
        torch.manual_seed(1712)
        num_nodes = data.num_nodes
        perm = torch.randperm(num_nodes)
        if train_size:
            train_idx = perm[:train_size]
            data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            data.train_mask[train_idx] = True
        if eval_size:
            val_idx = perm[train_size:train_size + eval_size]
            data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            data.val_mask[val_idx] = True
        if test_size:
            test_idx = perm[train_size + eval_size:train_size + eval_size + test_size]
            data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            data.test_mask[test_idx] = True
        return dataset, data, data, 'node'
    
    elif name == 'FLICKR':
        dataset = Flickr(root=os.path.join(path, 'Flickr'))
        data = dataset[0]
        task_type = 'node'
        return dataset, data, data, 'node'
    
    elif name == 'IMDB':
        dataset = IMDB(root=os.path.join(path, 'IMDB'))
        data = dataset[0]
        return dataset, data, data, 'node-het-movie'

    elif name == 'DBLP':
        dataset = DBLP(root=os.path.join(path, 'DBLP'))
        data = dataset[0]
        return dataset, data, data, 'node-het-author'

    elif name == 'AMINER':
        dataset = AMiner(root=os.path.join(path, 'AMiner'))
        data = dataset[0]
        return dataset, data, data, 'node-het-author'

    else:
        raise ValueError(f"Dataset '{name}' not supported.")

    if train_size and test_size:
        train_dataset = dataset[:train_size]
        test_dataset = dataset[train_size:train_size + test_size]
    else:
        train_dataset = dataset[:int(0.8 * len(dataset))]
        test_dataset = dataset[int(0.8 * len(dataset)):]

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return dataset, train_loader, test_loader, task_type


def eval_dataset(name, path='../data', eval_size=None, batch_size=32, seed=1309):
    name = name.upper()
    task_type = 'graph'

    if name in ['MUTAG', 'ENZYMES', 'PROTEINS']:
        dataset = TUDataset(os.path.join(path, 'TUDataset'), name=name)
        torch.manual_seed(seed)
        dataset = dataset.shuffle()
        eval_set = dataset[:eval_size] if eval_size else dataset[int(0.8 * len(dataset)):]
        eval_loader = DataLoader(eval_set, batch_size=batch_size)

    elif name == 'REDDIT-BINARY':
        dataset = TUDataset(os.path.join(path, 'TUDataset'), name=name, transform=ConstantX())
        torch.manual_seed(1712)
        dataset = dataset.shuffle()
        eval_set = dataset[:eval_size] if eval_size else dataset[int(0.8 * len(dataset)):]
        eval_loader = DataLoader(eval_set, batch_size=batch_size)
    elif name == 'ZINC':
        dataset = ZINC(root=os.path.join(path, 'ZINC'))
        torch.manual_seed(seed)
        dataset = dataset.shuffle()
        eval_set = dataset[:eval_size] if eval_size else dataset[int(0.8 * len(dataset)):]
        eval_loader = DataLoader(eval_set, batch_size=batch_size)

    elif name in ['CORA', 'CITESEER', 'PUBMED']:
        dataset = Planetoid(root=os.path.join(path, 'Planetoid'), name=name)
        eval_loader = dataset[0]
        task_type = 'node'

    elif name in ['CORNELL', 'WISCONSIN']:
        dataset = WebKB(root=os.path.join(path, 'WebKB'), name=name.lower())
        eval_loader = dataset[0]
        task_type = 'node'
        
    elif name in ['CHAMELEON', 'SQUIRREL', 'CROCODILE']:
        dataset = WikipediaNetwork(root=os.path.join(path, 'Wiki'),name=name.lower(), geom_gcn_preprocess=True if name != 'CROCODILE' else False) 
        eval_loader = dataset[0]
        task_type = 'node'
        
    elif name == 'FLICKR':
        dataset = Flickr(root='data/Flickr')
        eval_loader = dataset[0]
        task_type = 'node'
    
    elif name == 'IMDB':
        dataset = IMDB(root=os.path.join(path, 'IMDB'))
        eval_loader = dataset[0]
        task_type = 'node-het-movie'

    elif name == 'DBLP':
        dataset = DBLP(root=os.path.join(path, 'DBLP'))
        eval_loader = dataset[0]
        task_type = 'node-het-author'

    elif name == 'AMINER':
        dataset = AMiner(root=os.path.join(path, 'AMiner'))
        eval_loader = dataset[0]
        task_type = 'node-het-author'

    else:
        raise ValueError(f"Dataset '{name}' not supported.")

    return eval_loader, task_type

def random_split(data, train_ratio=0.6, val_ratio=0.2, seed=42):
    torch.manual_seed(seed)
    num_nodes = data.num_nodes
    perm = torch.randperm(num_nodes)

    train_end = int(train_ratio * num_nodes)
    val_end = train_end + int(val_ratio * num_nodes)

    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    data.train_mask[perm[:train_end]] = True
    data.val_mask[perm[train_end:val_end]] = True
    data.test_mask[perm[val_end:]] = True
    return data
