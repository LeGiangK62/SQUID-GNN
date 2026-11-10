import torch
import os
import json
from torch_geometric.datasets import TUDataset, ZINC, Planetoid, WikipediaNetwork, IMDB, DBLP, AMiner, Flickr, WebKB
from torch_geometric.loader import DataLoader
from utils import FixZINC, ConstantX, DegreeX, WebKBPreprocess
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.graphproppred import PygGraphPropPredDataset


EDGE_OGB_GRAPH_DATASETS = {
    "OGBG-MOLHIV": "ogbg-molhiv",
    "OGBG-MOLPCBA": "ogbg-molpcba",
    "OGBG-PPA": "ogbg-ppa",
    "OGBG-MOLTOX21": "ogbg-moltox21",
    "OGBG-MOLBACE": "ogbg-molbace",
    "OGBG-MOLBBBP": "ogbg-molbbbp",
    "OGBG-MOLCLINTOX": "ogbg-molclintox",
    "OGBG-MOLMUV": "ogbg-molmuv",
    "OGBG-MOLSIDER": "ogbg-molsider",
    "OGBG-MOLTOXCAST": "ogbg-moltoxcast",
    "OGBG-MOLESOL": "ogbg-molesol",
    "OGBG-MOLFREESOLV": "ogbg-molfreesolv",
    "OGBG-MOLLIPO": "ogbg-mollipo",
}


def _balanced_subset(dataset, indices, size):
    """Sample a balanced subset (50% label-0, 50% label-1) from given indices.
    If size is None, return the full split without balancing."""
    if size is None:
        return dataset[indices]
    labels = torch.tensor([dataset[int(i)].y.squeeze().item() for i in indices])
    idx_0 = indices[labels == 0]
    idx_1 = indices[labels == 1]
    half = size // 2
    n0 = min(half, len(idx_0))
    n1 = min(size - n0, len(idx_1))
    # If not enough of one class, fill with the other
    if n1 < size - n0:
        n0 = min(size - n1, len(idx_0))
    perm_0 = idx_0[torch.randperm(len(idx_0))[:n0]]
    perm_1 = idx_1[torch.randperm(len(idx_1))[:n1]]
    selected = torch.cat([perm_0, perm_1])
    return dataset[selected]


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

    elif name in EDGE_OGB_GRAPH_DATASETS:
        # Fix for PyTorch 2.6+ security restriction on legacy OGB files
        # We temporarily patch torch.load to allow weights_only=False
        _orig_load = torch.load
        def unsafe_load(*args, **kwargs):
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return _orig_load(*args, **kwargs)
        
        torch.load = unsafe_load
        
        try:
            if name == 'OGBN-ARXIV':
                dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=os.path.join(path, 'ogb'))
                data = dataset[0]
                split_idx = dataset.get_idx_split()
                
                # Convert OGB split indices to boolean masks for consistency with your other node datasets
                num_nodes = data.num_nodes
                data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
                data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
                data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
                
                data.train_mask[split_idx['train']] = True
                data.val_mask[split_idx['valid']] = True
                data.test_mask[split_idx['test']] = True
                
                # Flatten labels from (N, 1) to (N,) which is standard for PyG
                if data.y is not None:
                    data.y = data.y.squeeze()
                
                return dataset, data, data, 'node'

            # elif name == 'OGBG-MOLHIV':
            #     dataset = PygGraphPropPredDataset(name='ogbg-molhiv', root=os.path.join(path, 'ogb'))
            #     split_idx = dataset.get_idx_split()
                
            #     # Use OGB official splits
            #     train_dataset = dataset[split_idx['train']]
            #     test_dataset = dataset[split_idx['test']]
                
            #     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            #     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                
            #     return dataset, train_loader, test_loader, 'graph'
            
            elif name in EDGE_OGB_GRAPH_DATASETS:
                dataset = PygGraphPropPredDataset(
                    name=EDGE_OGB_GRAPH_DATASETS[name],
                    root=os.path.join(path, "ogb"),
                )
                split_idx = dataset.get_idx_split()

                train_dataset = _balanced_subset(dataset, split_idx["train"], train_size)
                # valid_dataset = _balanced_subset(dataset, split_idx["valid"], eval_size)
                test_dataset = _balanced_subset(dataset, split_idx["test"], test_size)

                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                # valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                return dataset, train_loader, test_loader, "graph"
            
            ## TODO: Add other dataset with edge

        finally:
            # Always restore the original secure torch.load
            torch.load = _orig_load

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

    elif name in EDGE_OGB_GRAPH_DATASETS:
        # Fix for PyTorch 2.6+ security restriction on legacy OGB files
        # We temporarily patch torch.load to allow weights_only=False
        _orig_load = torch.load
        def unsafe_load(*args, **kwargs):
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return _orig_load(*args, **kwargs)
        
        torch.load = unsafe_load
        
        try:
            dataset = PygGraphPropPredDataset(
                name=EDGE_OGB_GRAPH_DATASETS[name],
                root=os.path.join(path, "ogb"),
            )
            split_idx = dataset.get_idx_split()

            valid_dataset = dataset[split_idx["valid"]]

            if eval_size:
                valid_dataset = valid_dataset[:eval_size]

            eval_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

            task_type = "graph"

        finally:
            # Always restore the original secure torch.load
            torch.load = _orig_load

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


# ──────────────────────────────────────────────────────────────────────────────
# TUDataset graph classification: seed-driven, stratified, saved splits
# (used by main.py; load_dataset / eval_dataset above are left as they were for
#  the OGB, ZINC, node and hardware paths)
# ──────────────────────────────────────────────────────────────────────────────

TU_GRAPH_DATASETS = ('MUTAG', 'ENZYMES', 'PROTEINS', 'REDDIT-BINARY')


def _load_tu_dataset(name, path='../data'):
    """TUDataset in its original (unshuffled) order, so indices are stable."""
    name = name.upper()
    if name not in TU_GRAPH_DATASETS:
        raise ValueError(f"'{name}' is not a TUDataset graph-classification dataset {TU_GRAPH_DATASETS}.")
    transform = ConstantX() if name == 'REDDIT-BINARY' else None
    return TUDataset(os.path.join(path, 'TUDataset'), name=name, transform=transform)


def make_tu_splits(labels, seed, mode='holdout', n_folds=10, val_frac=0.1, test_frac=0.1):
    """
    Stratified train/val/test indices over the WHOLE dataset. The result depends only
    on (labels, seed, mode, n_folds, val_frac, test_frac) - never on global RNG state.

      holdout: test = `test_frac` of the dataset, val = `val_frac` of the dataset,
               train = the rest (default 80/10/10). One split ("fold 0").
      kfold:   Errica et al., ICLR 2020. StratifiedKFold(n_folds); each fold is the test
               set once; from the remaining graphs a stratified `val_frac` is held out
               as validation.

    Returns a list (one entry per fold) of {'fold', 'train', 'val', 'test'} with sorted indices.
    """
    from sklearn.model_selection import StratifiedKFold, train_test_split
    import numpy as np

    labels = np.asarray(labels).reshape(-1)
    all_idx = np.arange(len(labels))

    def split_off(idx, frac):
        rest, part = train_test_split(idx, test_size=frac, stratify=labels[idx],
                                      random_state=seed, shuffle=True)
        return rest, part

    folds = []
    if mode == 'holdout':
        if not (0 < test_frac < 1 and 0 < val_frac < 1 and val_frac + test_frac < 1):
            raise ValueError("holdout needs 0 < val_frac, test_frac and val_frac + test_frac < 1")
        rest, test = split_off(all_idx, test_frac)
        train, val = split_off(rest, val_frac / (1.0 - test_frac))
        folds.append((train, val, test))
    elif mode == 'kfold':
        if not 0 < val_frac < 1:
            raise ValueError("kfold needs 0 < val_frac < 1")
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for rest, test in skf.split(all_idx, labels):
            train, val = split_off(rest, val_frac)
            folds.append((train, val, test))
    else:
        raise ValueError(f"Unknown split mode '{mode}' (use 'holdout' or 'kfold').")

    return [
        {'fold': i,
         'train': sorted(int(j) for j in train),
         'val': sorted(int(j) for j in val),
         'test': sorted(int(j) for j in test)}
        for i, (train, val, test) in enumerate(folds)
    ]


def tu_split_file(name, seed, mode='holdout', n_folds=10, val_frac=0.1, test_frac=0.1, split_dir='../results/splits'):
    name = name.upper()
    if mode == 'holdout':
        tag = f"holdout_seed{seed}_val{val_frac}_test{test_frac}"
    else:
        tag = f"kfold{n_folds}_seed{seed}_val{val_frac}"
    return os.path.join(split_dir, f"{name.lower()}_{tag}.json")


def get_tu_splits(name, labels, seed, mode='holdout', n_folds=10, val_frac=0.1, test_frac=0.1,
                  split_dir='../results/splits'):
    """Load the split file for (dataset, seed, mode, ...) or create and save it. Returns (splits dict, path)."""
    name = name.upper()
    path = tu_split_file(name, seed, mode, n_folds, val_frac, test_frac, split_dir)
    config = {
        'dataset': name,
        'num_graphs': len(labels),
        'mode': mode,
        'seed': seed,
        'n_folds': n_folds if mode == 'kfold' else 1,
        'val_frac': val_frac,
        'test_frac': test_frac if mode == 'holdout' else None,
        'stratified': True,
    }
    if os.path.exists(path):
        with open(path, 'r') as f:
            splits = json.load(f)
        stored = {k: splits.get(k) for k in config}
        if stored != config:
            raise ValueError(f"Split file {path} was made with {stored}, but {config} was requested. "
                             f"Delete the file to regenerate it.")
        return splits, path

    splits = dict(config)
    splits['folds'] = make_tu_splits(labels, seed, mode, n_folds, val_frac, test_frac)
    os.makedirs(split_dir, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(splits, f)
    return splits, path


def load_tu_graph_split(name, path='../data', seed=0, mode='holdout', fold=0, n_folds=10,
                        val_frac=0.1, test_frac=0.1, batch_size=32, split_dir='../results/splits'):
    """
    TUDataset graph classification with a saved, stratified, seed-driven split.

    Returns: dataset, train_loader, val_loader, test_loader, info
      - the three subsets are disjoint and together cover the whole dataset
      - the train loader is reshuffled every epoch with its own generator seeded by `seed`
      - info: {'split_file', 'fold', 'n_train', 'n_val', 'n_test', 'train', 'val', 'test'}
    """
    dataset = _load_tu_dataset(name, path)
    labels = dataset.y.view(-1).tolist()
    splits, split_path = get_tu_splits(name, labels, seed, mode, n_folds, val_frac, test_frac, split_dir)

    if not 0 <= fold < len(splits['folds']):
        raise ValueError(f"fold {fold} is out of range: this split has {len(splits['folds'])} fold(s).")
    idx = splits['folds'][fold]

    train_dataset = dataset[torch.tensor(idx['train'])]
    val_dataset = dataset[torch.tensor(idx['val'])]
    test_dataset = dataset[torch.tensor(idx['test'])]

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    info = {
        'split_file': split_path,
        'fold': fold,
        'n_train': len(idx['train']),
        'n_val': len(idx['val']),
        'n_test': len(idx['test']),
        'train': idx['train'],
        'val': idx['val'],
        'test': idx['test'],
    }
    return dataset, train_loader, val_loader, test_loader, info
