import os
import copy
import json
import random
import hashlib

# Needed by torch.use_deterministic_algorithms on CUDA (see main_tu); must be set before CUDA starts.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import matplotlib.pyplot as plt
import pennylane as qml
import argparse
from torch import nn, optim
import numpy as np


from utils import train_graph, test_graph, train_node, test_node, save_checkpoint
from data import load_dataset, eval_dataset, random_split
from data import load_tu_graph_split, TU_GRAPH_DATASETS
from model import QGNNGraphClassifier_Batched as QGNNGraphClassifier, QGNNNodeClassifier_Batched as QGNNNodeClassifier
from test import HandcraftGNN, HandcraftGNN_NodeClassification

from datetime import datetime
import time


timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")


result_dir = os.path.join('../results')
os.makedirs(result_dir, exist_ok=True)
os.makedirs(os.path.join(result_dir, 'fig'), exist_ok=True)
os.makedirs(os.path.join(result_dir, 'log'), exist_ok=True)
os.makedirs(os.path.join(result_dir, 'model'), exist_ok=True)
os.makedirs(os.path.join(result_dir, 'train_plot'), exist_ok=True)

param_file = os.path.join(result_dir, 'log', f"{timestamp}_model_parameters.txt")
grad_file = os.path.join(result_dir, 'log', f"{timestamp}_model_gradients.txt")


def get_args():
    parser = argparse.ArgumentParser(description="Train QGNN on graph data")
    parser.add_argument('--dataset', type=str, default='MUTAG', help='Dataset name (e.g., MUTAG, ENZYMES, CORA)')
    # Sizes for the legacy paths only (node classification, ZINC, OGB through main.py).
    # TUDataset graph classification uses the whole dataset and ignores them (with a warning).
    parser.add_argument('--train_size', type=int, default=None, help='legacy paths only (default 100)')
    parser.add_argument('--eval_size', type=int, default=None, help='legacy paths only (default 150)')
    parser.add_argument('--test_size', type=int, default=None, help='legacy paths only (default 50)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-2)
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--node_qubit', type=int, default=3)
    parser.add_argument('--num_gnn_layers', type=int, default=2)
    parser.add_argument('--num_ent_layers', type=int, default=1)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--seed', type=int, default=1712,
                        help='Seeds torch / numpy / python random. For TUDataset graph classification it also '
                             'defines the data split, the batch order and the model initialisation.')
    parser.add_argument('--task', type=str, default='graph', choices=['graph', 'node'], help='graph or node classification')

    # Evaluation protocol for TUDataset graph classification (MUTAG, ENZYMES, PROTEINS, REDDIT-BINARY)
    parser.add_argument('--split_mode', type=str, default='holdout', choices=['holdout', 'kfold'],
                        help='holdout: stratified train/val/test (default 80/10/10). '
                             'kfold: stratified k-fold test sets, validation held out from the rest (Errica et al. 2020).')
    parser.add_argument('--val_frac', type=float, default=0.1,
                        help='holdout: validation fraction of the whole dataset. kfold: fraction of the non-test graphs.')
    parser.add_argument('--test_frac', type=float, default=0.1, help='holdout only: test fraction of the whole dataset')
    parser.add_argument('--folds', type=int, default=10, help='kfold only: number of folds')
    parser.add_argument('--fold', type=int, default=0, help='kfold only: which fold a single run uses as test set')
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping: stop after this many epochs without a better validation metric')
    parser.add_argument('--num_seeds', type=int, default=5,
                        help='--results in holdout mode: number of runs, with seeds seed, seed+1, ...')


    # Debug options
    parser.add_argument('--pre_train', type=str, default=None, help='Load the pre-trained model (timestamp as name)')
    parser.add_argument('--continue_train', action='store_true', help='Continue training from pre-trained model')
    parser.add_argument('--plot', action='store_true', help='Enable plotting')
    parser.add_argument('--save_model', action='store_true',
                        help='Save the model. TUDataset graph path: the checkpoint of the best-validation epoch.')
    parser.add_argument('--gradient', action='store_true', help='Enable gradient saving')
    parser.add_argument('--results', action='store_true',
                        help='TUDataset graph path: train a fresh model per seed (holdout) or per fold (kfold) and '
                             'report mean/std of the test metric at the best-validation epoch, saved as JSON. '
                             'Legacy paths: old behaviour.')
    parser.add_argument('--no_Edge', action='store_true', default=False, help='Omit edge attributes')
    parser.add_argument('--count_params', action='store_true', help='Print number of model parameters and exit')
    parser.add_argument('--criterion', type=str, default='crossentropy',
                        choices=['crossentropy', 'MSE', 'BCE', 'L1', 'NLL'],
                        help="Which loss function to train model")
    
    # For switching between models
    parser.add_argument('--model', type=str, default='qgnn', 
                        choices=['qgnn', 'handcraft', 'gin', 'gcn', 'gat', 'sage', 'trans', 'hqgnn'],
                        help="Which model to run"
                        )
    parser.add_argument('--graphlet_size', type=int, default=10)
    parser.add_argument('--new_agg', action='store_true',
                        help='qgnn model (graph and node): use the new AGG circuit (W_pre -> [Rot + CNOT] per neighbor -> W_post)')
    parser.add_argument('--no_pad', action='store_true',
                        help='qgnn model (graph and node), needs --new_agg: nodes with fewer neighbors than the subgraph size '
                             'run a smaller circuit instead of being zero-padded')
    parser.add_argument('--norm', type=str, default='layer', choices=['layer', 'batch', 'none'],
                        help='qgnn model (graph and node): norm applied to the node update before the residual sum '
                             '(layer = LayerNorm(2), batch = BatchNorm1d(2), none = no norm)')

    
    return parser.parse_args()


def make_quantum_config(args):
    """Quantum device + PQC weight shapes for the current arguments (shared by every path)."""
    aux_qubit = 0 if args.new_agg else 1  # the new AGG circuit has no auxiliary qubit
    args.node_qubit = args.graphlet_size
    edge_qubit = args.node_qubit - 1
    n_qubits = args.node_qubit + edge_qubit
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_dev = qml.device("default.qubit", wires=n_qubits + aux_qubit) # number of ancilla qubits
    print(f'Quantum device: {n_qubits} qubit - {q_dev}')

    # PQC weight shape settings
    w_shapes_dict = {
        # 'spreadlayer': (0, n_qubits, 1),
        # Old
        # 'strong': (2, args.num_ent_layers, 3, 3), # 3
        # # 'strong': (3, args.num_ent_layers, 2, 3), # 2
        # 'inits': (1, 4),
        # 'update': (1, args.num_ent_layers, 3, 3), # (1, args.num_ent_layers, 2, 3)
        # NEW
        # 'inits': (1, 2), # New
        # 'strong': (1, args.num_ent_layers, 2, 3), # strong ent
        'inits': (args.num_ent_layers, 4), # Custom 14
        'strong': (args.num_ent_layers, 4), # custom 14
        'update': (edge_qubit, args.num_ent_layers, 2 + aux_qubit, 3), 
        # 'update': (args.graphlet_size, args.num_ent_layers-1, 4, 3),
        # 'twodesign': (0, args.num_ent_layers, 1, 2)
    }
    if args.new_agg and args.model == 'qgnn':
        # New AGG circuit (graph and node qgnn models): no `update`, no auxiliary qubit
        w_shapes_dict = {
            'inits': (args.num_ent_layers, 4),
            'strong': (args.num_ent_layers, 4),
            'w_pre': (3,),   # Rot on center, before AGG
            'agg': (3,),     # Rot on each neighbor, shared
            'w_post': (3,),  # Rot on center, after AGG
        }
    return q_dev, w_shapes_dict, device


def build_model(args, task_type, het_node_type, dataset, q_dev, w_shapes_dict, device,
                node_input_dim, edge_input_dim, num_classes):
    """Construct the model selected by --model for a graph or node task (not yet moved to `device`)."""
    if task_type == 'graph':
        if args.model == 'qgnn':
            model = QGNNGraphClassifier(
                q_dev=q_dev,
                w_shapes=w_shapes_dict,
                hidden_dim=args.hidden_channels,
                node_input_dim=node_input_dim,
                edge_input_dim=edge_input_dim,
                graphlet_size=args.node_qubit,
                hop_neighbor=args.num_gnn_layers,
                num_classes=num_classes,
                one_hot=0,
                new_agg=args.new_agg,
                no_pad=args.no_pad,
                norm=args.norm
            )
        elif args.model == 'hqgnn':
            from baseline import HQGNN_Graph
            model = HQGNN_Graph(
                in_channels=node_input_dim,
                hidden_channels=args.hidden_channels,
                out_channels=num_classes,
                num_layers=args.num_gnn_layers,
            )
        elif args.model == 'handcraft':
            model = HandcraftGNN(
                q_dev=q_dev,
                w_shapes=w_shapes_dict,
                node_input_dim=node_input_dim,
                edge_input_dim=edge_input_dim,
                graphlet_size=args.graphlet_size,
                hop_neighbor=args.num_gnn_layers,
                num_classes=num_classes,
                one_hot=0
            )
        elif args.model == 'gin':
            from baseline import GIN_Graph
            model = GIN_Graph(
                in_channels=node_input_dim,
                hidden_channels=args.hidden_channels,
                out_channels=num_classes,
                num_layers=args.num_gnn_layers,
            )
        elif args.model == 'gcn':
            from baseline import GCN_Graph
            model = GCN_Graph(
                in_channels=node_input_dim,
                hidden_channels=args.hidden_channels,
                out_channels=num_classes,
                num_layers=args.num_gnn_layers,
            )
        elif args.model == 'gat':
            from baseline import GAT_Graph
            model = GAT_Graph(
                in_channels=node_input_dim,
                hidden_channels=args.hidden_channels//8,    # heads * hidden
                out_channels=num_classes,
                num_layers=args.num_gnn_layers,
                heads=8,
            )
        elif args.model == 'sage':
            from baseline import GraphSAGE_Graph
            model = GraphSAGE_Graph(
                in_channels=node_input_dim,
                hidden_channels=args.hidden_channels,
                out_channels=num_classes,
                num_layers=args.num_gnn_layers
            )
        elif args.model == 'trans':
            from baseline import Transformer_Graph
            model = Transformer_Graph(
                in_channels=node_input_dim,
                hidden_channels=args.hidden_channels//8,    # heads * hidden
                out_channels=num_classes,
                num_layers=args.num_gnn_layers,
                heads=8,
            )
        else:
            raise ValueError(f"Unsupported model for graph task: {args.model}")
    elif task_type == 'node':
        data = dataset[0].to(device)
        if het_node_type is not None:
            if args.model in ['gin', 'sage', 'gcn', 'trans']:
                from baseline import HeteroGNN_Node
                model = HeteroGNN_Node(
                    name=args.model,
                    node_name=het_node_type,
                    metadata=data.metadata(),               
                    in_channels=node_input_dim,
                    hidden_channels=args.hidden_channels,
                    out_channels=num_classes,
                    num_layers=args.num_gnn_layers,
                    heads=8  # used only if name='trans'
                )
        else:
            if args.model == 'qgnn':
                model = QGNNNodeClassifier(
                    q_dev=q_dev,
                    w_shapes=w_shapes_dict,
                    hidden_dim=args.hidden_channels,
                    node_input_dim=node_input_dim,
                    edge_input_dim=edge_input_dim,
                    graphlet_size=args.node_qubit,
                    hop_neighbor=args.num_gnn_layers,
                    num_classes=num_classes,
                    one_hot=0,
                    new_agg=args.new_agg,
                    no_pad=args.no_pad,
                    norm=args.norm
                )
            elif args.model == 'handcraft':
                model = HandcraftGNN_NodeClassification(
                    q_dev=q_dev,
                    w_shapes=w_shapes_dict,
                    node_input_dim=node_input_dim,
                    edge_input_dim=edge_input_dim,
                    graphlet_size=args.graphlet_size,
                    hop_neighbor=args.num_gnn_layers,
                    num_classes=num_classes,
                    one_hot=0
                )
            elif args.model == 'hqgnn':
                from baseline import HQGNN_Node
                model = HQGNN_Node(
                    in_channels=node_input_dim,
                    hidden_channels=args.hidden_channels,
                    out_channels=num_classes,
                    num_layers=args.num_gnn_layers,
                )
            elif args.model in ['gin', 'sage', 'gcn', 'trans']:
                from baseline import GNN_Node
                model = GNN_Node(
                    name=args.model,               
                    in_channels=node_input_dim,
                    hidden_channels=args.hidden_channels,
                    out_channels=num_classes,
                    num_layers=args.num_gnn_layers,
                    heads=8  # used only if name='trans'
                )
            elif args.model == 'gat':
                from baseline import GAT_Node
                model = GAT_Node(
                    in_channels=node_input_dim,
                    hidden_channels=8,    # heads * hidden
                    out_channels=num_classes,
                    num_layers=args.num_gnn_layers,
                    heads=8,
                )
            # elif args.model == 'trans':
            #     from baseline import Transformer_Node
            #     model = Transformer_Node(
            #         in_channels=node_input_dim,
            #         hidden_channels=args.hidden_channels//8,    # heads * hidden
            #         out_channels=num_classes,
            #         num_layers=args.num_gnn_layers,
            #         heads=8,
            #     )
            else:
                raise ValueError(f"Unsupported model for node task: {args.model}")
    else:
        raise ValueError("Unsupported task type")
    return model


def print_param_count(model, args):
    quantum_params = 0
    classical_params = 0
    for module in model.modules():
        if isinstance(module, qml.qnn.TorchLayer):
            quantum_params += sum(p.numel() for p in module.parameters())
        elif list(module.children()) == []:  # leaf module only
            classical_params += sum(p.numel() for p in module.parameters())
    total = quantum_params + classical_params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model}")
    print(f"Total parameters:     {total}")
    print(f"  Quantum parameters: {quantum_params}")
    print(f"  Classical parameters: {classical_params}")
    print(f"Trainable parameters: {trainable}")


def make_criterion(args):
    if args.criterion == 'crossentropy':
        criterion = nn.CrossEntropyLoss()
    elif args.criterion == 'MSE':
        criterion = nn.MSELoss()
    elif args.criterion == 'BCE':
        criterion = nn.BCEWithLogitsLoss()
    elif args.criterion == 'L1':
        criterion = nn.L1Loss()
    elif args.criterion == 'NLL':
        criterion  = nn.NLLLoss()
    else:
        raise ValueError(f"Unssuported loss function")
    return criterion


def main_legacy(args):
    """Original pipeline, unchanged: node classification, ZINC and OGB datasets run through main.py."""
    # These three flags only exist for this path; same defaults as before.
    if args.train_size is None: args.train_size = 100
    if args.eval_size is None: args.eval_size = 150
    if args.test_size is None: args.test_size = 50

    q_dev, w_shapes_dict, device = make_quantum_config(args)

    # Load dataset
    dataset, train_loader, test_loader, task_type = load_dataset(
        name=args.dataset,
        path='../data',
        train_size=args.train_size,
        test_size=args.test_size,
        eval_size=args.eval_size,
        batch_size=args.batch_size
    )
    
    if len(task_type.split('-')) == 3:
        task_type, _, het_node_type = task_type.split('-')
    else:
        het_node_type = None
    
    result_base = f"{timestamp}_{args.model}_{args.graphlet_size}_{args.epochs}_{args.lr}"
    plot_train_path = os.path.join(result_dir, 'fig', f"{args.dataset.lower()}_plot_{result_base}_train.png")
    npz_path = os.path.join(result_dir, 'train_plot', f"{args.dataset.lower()}_data_{result_base}.npz")
    model_save = os.path.join(result_dir, 'model', f"{args.dataset.lower()}_model_{result_base}.pt")

    # if task_type != 'graph':
    #     raise NotImplementedError("Node classification support is not implemented yet.")
 
    # Model metadata
    if het_node_type is None:
        node_input_dim = dataset[0].x.shape[1] if dataset[0].x is not None else 0
        if args.no_Edge:
            edge_input_dim = 0
        else:
            edge_input_dim = dataset[0].edge_attr.shape[1] if dataset[0].edge_attr is not None else 0
        num_classes = dataset.num_classes
    else:
        node_input_dim = dataset[0][het_node_type].x.shape[1] if dataset[0][het_node_type].x is not None else 0
        edge_input_dim = 0
        for edge_type in dataset[0].edge_types:
            edge_attr = dataset[0][edge_type].get('edge_attr', None)
            if not args.no_Edge and edge_attr is not None:
                edge_input_dim = edge_attr.shape[1]
                break  # take the first found
        num_classes = int(dataset[0][het_node_type].y.max()) + 1
    # Model init
    if args.dataset == 'ogbg-molhiv': num_classes = 1
    model = build_model(args, task_type, het_node_type, dataset, q_dev, w_shapes_dict, device,
                        node_input_dim, edge_input_dim, num_classes)
    
    model = model.to(device)

    if args.count_params:
        print_param_count(model, args)
        return

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    criterion = make_criterion(args)
    # criterion = nn.NLLLoss() ## MUTAG

    ## Note: For debugging purposes, you can uncomment the following lines to print model details. 
    # ##
    # print("=" * 50)
    # print(f"Training on dataset: {args.dataset.upper()}")
    # print(f"Node feature dimension: {node_input_dim}")
    # print(f"Edge feature dimension: {edge_input_dim}")
    # print(f"Number of classes: {num_classes}")
    # print(f"Number of training samples: {len(train_loader.dataset)}")
    # print(f"Number of testing samples: {len(test_loader.dataset)}")
    # print(f"QGNN layers: {args.num_gnn_layers}")
    # print(f"Entangling layers per PQC: {args.num_ent_layers}")
    # print(f"Total qubits: {n_qubits} (Node qubits: {args.node_qubit}, Edge qubits: {edge_qubit})")
    # print(f"Epochs: {args.epochs}")
    # print(f"Batch size: {args.batch_size}")
    # print(f"Learning rate: {args.lr}")
    # print("=" * 50)

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    # Training loop
    if args.gradient:
        string = "="*10 + f"{timestamp}_{args.model}_{args.graphlet_size}_{args.dataset.lower()}_{args.epochs}epochs_lr{args.lr}_{args.gamma}over{args.step_size}" + "="*10
        with open(param_file, "w") as f_param:
            f_param.write(string + "\n")
        with open(grad_file, "w") as f_grad:
            f_grad.write(string + "\n")
        
    start = time.time()
    step_plot = args.epochs // 10 if args.epochs > 10 else 1
    
    # early_stopping = EarlyStopping(patience=10, save_path=model_save)
    
    print(f"\n ===={timestamp}==== ")
    
    if args.pre_train is not None:
        pre_trained_path = os.path.join(result_dir, 'model', f"{args.dataset.lower()}_model_{args.pre_train}.pt")
        checkpoint = torch.load(pre_trained_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        pre_train_npz_path = os.path.join(result_dir, 'train_plot', f"{args.dataset.lower()}_data_{args.pre_train}.npz")
        pre_train_data = np.load(pre_train_npz_path)
        pre_train_epoch = pre_train_data['epoch'].shape[0]          
        train_losses = pre_train_data['train_losses'].tolist()
        test_losses = pre_train_data['test_losses'].tolist()
        train_accs = pre_train_data['train_accs'].tolist()
        test_accs = pre_train_data['test_accs'].tolist()
        print(f"Pre-trained model loaded from {pre_trained_path} with {pre_train_epoch} epochs.")
        if not args.continue_train: 
            model.eval()
            print("Skip training...")
        else:
            print(f"Continuing training model {args.model} with {args.graphlet_size} graphlet size with {args.epochs} epochs, "
            f"learning rate {args.lr}, step size {args.step_size}, and gamma {args.gamma}.")
    else: 
        print(f"Training model {args.model} on {args.dataset} - {num_classes} classes with {args.graphlet_size} graphlet size with {args.epochs} epochs, "
            f"learning rate {args.lr}, step size {args.step_size}, and gamma {args.gamma}.")
            
        
        
        
    # print(f"Training model {args.model} on {args.dataset} with {args.graphlet_size} graphlet size with {args.epochs} epochs, "
    #       f"learning rate {args.lr}, step size {args.step_size}, and gamma {args.gamma}.")
    if args.continue_train or args.pre_train is None:
        if task_type == 'graph':
            for epoch in range(args.epochs):
                train_graph(model, optimizer, train_loader, criterion, device)
                train_loss, train_acc, f1_train = test_graph(model, train_loader, criterion, device, num_classes)
                test_loss, test_acc, f1_test = test_graph(model, test_loader, criterion, device, num_classes)
                scheduler.step()
                if args.save_model:
                    # early_stopping(-avg_test_sinr, model)
                    save_checkpoint(model, optimizer, model_save)
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                train_accs.append(train_acc)
                test_accs.append(test_acc)
                np.savez_compressed(
                    npz_path, 
                    epoch=np.arange(1, epoch+2),
                    train_losses=np.array(train_losses),
                    test_losses=np.array(test_losses),
                    train_accs=np.array(train_accs),
                    test_accs=np.array(test_accs),
                )
                ############
                if args.gradient:
                    # === Write model parameters to file ===
                    with open(param_file, "a") as f_param:
                        f_param.write("="*40 + f" Epoch {epoch} " + "="*40 + "\n")
                        for name, param in model.named_parameters():
                            f_param.write(f"{name}:\n{param.data.cpu().numpy()}\n\n")

                    # === Write gradients to separate file ===
                    with open(grad_file, "a") as f_grad:
                        f_grad.write("="*40 + f" Epoch {epoch} " + "="*40 + "\n")
                        for name, param in model.named_parameters():
                            if param.requires_grad:
                                if param.grad is None:
                                    f_grad.write(f"{name}: No gradient (None)\n")
                                else:
                                    grad = param.grad.cpu().numpy()
                                    f_grad.write(f"{name}:\n{grad}\n\n")
                ############
                if epoch % step_plot == 0:
                    print(f"Epoch {epoch+1:02d}/{args.epochs+1:02d} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                        f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
        elif task_type=='node':  # node task
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min',                           
                factor=args.gamma,      # Multiplies LR by this factor (e.g., 0.5)
                patience=args.epochs//10,# Wait this many epochs without improvement
                # verbose=True                            # Print updates
            )
            for epoch in range(args.epochs):
                train_loss = train_node(model, optimizer, train_loader, criterion, device, het_node_type)
                test_metrics = test_node(model, test_loader, criterion, device, het_node_type)
                train_losses.append(test_metrics['train']['loss'])
                test_losses.append(test_metrics['test']['loss'])
                train_accs.append(test_metrics['train']['acc'])
                test_accs.append(test_metrics['test']['acc'])
                
                if args.save_model:
                    # early_stopping(test_losses[-1], model)
                    save_checkpoint(model, optimizer, model_save)
                np.savez_compressed(
                    npz_path, 
                    epoch=np.arange(1, epoch+2),
                    train_losses=np.array(train_losses),
                    test_losses=np.array(test_losses),
                    train_accs=np.array(train_accs),
                    test_accs=np.array(test_accs),
                )
                                        
                scheduler.step(test_metrics['val']['loss'])
                if epoch % step_plot == 0:
                    print(f"Epoch {epoch+1:02d}/{args.epochs+1:02d} | Train Loss: {train_loss:.4f} |" +
                        f"Train Acc: {test_metrics['train']['acc']:.4f} | "
                        f"Val Acc: {test_metrics['val']['acc']:.4f} | Test Acc: {test_metrics['test']['acc']:.4f}")
    if args.save_model:
            print(f"Model checkpoint saved to {model_save}")
    end = time.time()
    print(f"Total execution time: {end - start:.6f} seconds")
    if args.plot:
        if args.pre_train is None:
            pre_train_epoch = 0
        total_epoch = args.epochs + pre_train_epoch
        epochs_range = range(1, total_epoch + 1)

        plt.figure(figsize=(10, 5))
        plt.suptitle(f"{args.model.upper()} on {args.dataset.upper()}", fontsize=14)
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, train_losses, label="Train Loss")
        plt.plot(epochs_range, test_losses, label="Test Loss")
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, train_accs, label="Train Acc")
        plt.plot(epochs_range, test_accs, label="Test Acc")
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        # plt.tight_layout()
        plt.tight_layout(rect=[0, 0, 1, 0.95]) 
        # plot_path = f"plot_{args.model}_{args.graphlet_size}_{args.dataset.lower()}_{args.epochs}epochs_lr{args.lr}_{args.gamma}over{args.step_size}.png"
        # plot_path = f"plot_{timestamp}_{args.model}_{args.graphlet_size}_{args.dataset.lower()}_{args.epochs}epochs_lr{args.lr}_{args.gamma}over{args.step_size}.png"
        plt.savefig(plot_train_path, dpi=300)
        
    if args.results:
        accuracies = []
        num_runs = 100  
        for each in range(num_runs):
            eval_loader, _ = eval_dataset(
                name=args.dataset,
                path='../data',
                eval_size=args.eval_size,
                batch_size=args.batch_size,
                seed=args.seed+each
            )
            if task_type == 'graph':
                _, eval_acc, _ = test_graph(model, eval_loader, criterion, device, num_classes)
            elif task_type == 'node':
                eval_loader = random_split(eval_loader, train_ratio=0.6, val_ratio=0.2, seed=args.seed+each)
                eval_metrics = test_node(model, eval_loader, criterion, device, het_node_type)
                eval_acc = eval_metrics['val']['acc']
            else:
                raise ValueError(f"Unsupported task: {task_type}")
            
            accuracies.append(eval_acc)

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies, ddof=1)  # unbiased std deviation

        print(f"{args.model} Mean Accuracy: {mean_acc:.4f} ± {std_acc:.3f}")


# ──────────────────────────────────────────────────────────────────────────────
# TUDataset graph classification: train / validation / test protocol
#   - split: stratified, whole dataset, defined by (dataset, seed, mode, fold), saved to results/splits/
#   - model selection: best validation accuracy (ties -> lower validation loss), early stopping
#   - reported number: test accuracy at the best-validation epoch
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed):
    """Seed python, numpy and torch (CPU + CUDA): model init, dropout, neighbor / star sampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _is_better(val_acc, val_loss, best):
    """Validation-only selection rule: higher accuracy, ties broken by lower loss."""
    if best is None:
        return True
    if val_acc != best['val_acc']:
        return val_acc > best['val_acc']
    return val_loss < best['val_loss']


def run_tu_graph(args, seed, fold, q_dev, w_shapes_dict, device):
    """
    One training run on a TUDataset with the split (dataset, seed, split_mode, fold).
    Returns a dict with the metrics at the best-validation epoch (None for --count_params).
    """
    set_seed(seed)

    dataset, train_loader, val_loader, test_loader, split = load_tu_graph_split(
        name=args.dataset,
        path='../data',
        seed=seed,
        mode=args.split_mode,
        fold=fold,
        n_folds=args.folds,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        batch_size=args.batch_size,
        split_dir=os.path.join(result_dir, 'splits'),
    )

    node_input_dim = dataset[0].x.shape[1] if dataset[0].x is not None else 0
    if args.no_Edge:
        edge_input_dim = 0
    else:
        edge_input_dim = dataset[0].edge_attr.shape[1] if dataset[0].edge_attr is not None else 0
    num_classes = dataset.num_classes

    model = build_model(args, 'graph', None, dataset, q_dev, w_shapes_dict, device,
                        node_input_dim, edge_input_dim, num_classes)
    model = model.to(device)

    if args.count_params:
        print_param_count(model, args)
        return None

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    criterion = make_criterion(args)

    run_tag = f"seed{seed}" + (f"_fold{fold}" if args.split_mode == 'kfold' else "")
    result_base = f"{timestamp}_{args.model}_{args.graphlet_size}_{args.epochs}_{args.lr}_{run_tag}"
    plot_train_path = os.path.join(result_dir, 'fig', f"{args.dataset.lower()}_plot_{result_base}_train.png")
    npz_path = os.path.join(result_dir, 'train_plot', f"{args.dataset.lower()}_data_{result_base}.npz")
    model_save = os.path.join(result_dir, 'model', f"{args.dataset.lower()}_model_{result_base}.pt")

    print(f"\n ===={timestamp} | {run_tag}==== ")
    print(f"Training model {args.model} on {args.dataset} - {num_classes} classes, graphlet size {args.graphlet_size}, "
          f"up to {args.epochs} epochs (patience {args.patience}), lr {args.lr}, step size {args.step_size}, gamma {args.gamma}.")
    print(f"Split: {args.split_mode}" + (f" fold {fold}/{args.folds}" if args.split_mode == 'kfold' else "")
          + f", seed {seed} | train {split['n_train']} / val {split['n_val']} / test {split['n_test']} | {split['split_file']}")

    if args.gradient:
        header = "=" * 10 + f"{timestamp}_{args.model}_{args.graphlet_size}_{args.dataset.lower()}_{run_tag}" + "=" * 10
        with open(param_file, "a") as f_param:
            f_param.write(header + "\n")
        with open(grad_file, "a") as f_grad:
            f_grad.write(header + "\n")

    hist = {k: [] for k in ('train_losses', 'val_losses', 'test_losses', 'train_accs', 'val_accs', 'test_accs')}
    best = None           # metrics at the best-validation epoch
    best_state = None     # model weights at the best-validation epoch
    early_stopped = False
    step_plot = args.epochs // 10 if args.epochs > 10 else 1
    start = time.time()

    for epoch in range(args.epochs):
        train_graph(model, optimizer, train_loader, criterion, device)
        train_loss, train_acc, _ = test_graph(model, train_loader, criterion, device, num_classes)
        val_loss, val_acc, _ = test_graph(model, val_loader, criterion, device, num_classes)
        test_loss, test_acc, _ = test_graph(model, test_loader, criterion, device, num_classes)
        scheduler.step()   # epoch-based StepLR: uses no data split

        for key, value in zip(hist, (train_loss, val_loss, test_loss, train_acc, val_acc, test_acc)):
            hist[key].append(value)

        # Model selection: validation only
        if _is_better(val_acc, val_loss, best):
            best = {
                'best_epoch': epoch + 1,
                'train_acc': train_acc, 'train_loss': train_loss,
                'val_acc': val_acc, 'val_loss': val_loss,
                'test_acc': test_acc, 'test_loss': test_loss,
            }
            best_state = copy.deepcopy(model.state_dict())
            if args.save_model:
                save_checkpoint(model, optimizer, model_save)

        np.savez_compressed(
            npz_path,
            epoch=np.arange(1, epoch + 2),
            train_losses=np.array(hist['train_losses']),
            val_losses=np.array(hist['val_losses']),
            test_losses=np.array(hist['test_losses']),
            train_accs=np.array(hist['train_accs']),
            val_accs=np.array(hist['val_accs']),
            test_accs=np.array(hist['test_accs']),
            best_val_epoch=best['best_epoch'],
            test_acc_at_best_val=best['test_acc'],
        )

        if args.gradient:
            with open(param_file, "a") as f_param:
                f_param.write("=" * 40 + f" Epoch {epoch} " + "=" * 40 + "\n")
                for name, param in model.named_parameters():
                    f_param.write(f"{name}:\n{param.data.cpu().numpy()}\n\n")
            with open(grad_file, "a") as f_grad:
                f_grad.write("=" * 40 + f" Epoch {epoch} " + "=" * 40 + "\n")
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        if param.grad is None:
                            f_grad.write(f"{name}: No gradient (None)\n")
                        else:
                            f_grad.write(f"{name}:\n{param.grad.cpu().numpy()}\n\n")

        if epoch % step_plot == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

        if (epoch + 1) - best['best_epoch'] >= args.patience:
            early_stopped = True
            print(f"Early stopping at epoch {epoch+1}: no better validation metric for {args.patience} epochs.")
            break

    epochs_run = len(hist['train_losses'])
    model.load_state_dict(best_state)   # leave the model at its best-validation weights
    print(f"Total execution time: {time.time() - start:.6f} seconds")
    print(f"[{run_tag}] best validation epoch {best['best_epoch']} (val acc {best['val_acc']:.4f}, val loss {best['val_loss']:.4f}) "
          f"-> test acc at that epoch: {best['test_acc']:.4f}")
    if args.save_model:
        print(f"Best-validation checkpoint saved to {model_save}")

    if args.plot:
        epochs_range = range(1, epochs_run + 1)
        plt.figure(figsize=(10, 5))
        plt.suptitle(f"{args.model.upper()} on {args.dataset.upper()} ({run_tag})", fontsize=14)
        for pos, (key, title) in enumerate((('losses', 'Loss'), ('accs', 'Accuracy')), start=1):
            plt.subplot(1, 2, pos)
            plt.plot(epochs_range, hist[f'train_{key}'], label=f"Train {title}")
            plt.plot(epochs_range, hist[f'val_{key}'], label=f"Val {title}")
            plt.plot(epochs_range, hist[f'test_{key}'], label=f"Test {title}")
            plt.axvline(best['best_epoch'], color='gray', linestyle='--', linewidth=0.8, label="Best val epoch")
            plt.title(title)
            plt.xlabel("Epoch")
            plt.ylabel(title)
            plt.legend()
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(plot_train_path, dpi=300)
        plt.close()

    summary = {'seed': seed, 'fold': fold if args.split_mode == 'kfold' else None}
    summary.update(best)
    summary.update({
        'epochs_run': epochs_run,
        'early_stopped': early_stopped,
        'n_train': split['n_train'], 'n_val': split['n_val'], 'n_test': split['n_test'],
        'split_file': split['split_file'],
        'curves_file': npz_path,
        'checkpoint': model_save if args.save_model else None,
    })
    return summary


def main_tu(args):
    """TUDataset graph classification: single run, or --results over seeds (holdout) / folds (kfold)."""
    # Flags that have no meaning on this path: say so instead of silently ignoring them.
    for flag in ('train_size', 'test_size', 'eval_size'):
        if getattr(args, flag) is not None:
            print(f"WARNING: --{flag} is ignored for {args.dataset.upper()}: the whole dataset is split with "
                  f"--split_mode / --val_frac / --test_frac / --folds.")
    if args.pre_train is not None or args.continue_train:
        raise SystemExit("--pre_train / --continue_train are not supported for TUDataset graph classification: "
                         "every run trains a fresh model on its own train split and is selected on validation.")
    if args.split_mode == 'holdout' and args.fold != 0:
        raise SystemExit("--fold is only used with --split_mode kfold.")
    if args.split_mode == 'kfold' and not 0 <= args.fold < args.folds:
        raise SystemExit(f"--fold must be in [0, {args.folds - 1}].")

    # Same seed -> same curves, also on GPU (without this, CUDA runs drift apart after the first step).
    # warn_only: an operation without a deterministic implementation warns instead of failing.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False

    q_dev, w_shapes_dict, device = make_quantum_config(args)

    if not args.results:
        run_tu_graph(args, args.seed, args.fold, q_dev, w_shapes_dict, device)
        return

    # --results: a fresh model per seed (holdout) or per fold (kfold)
    if args.split_mode == 'holdout':
        runs = [(args.seed + i, 0) for i in range(args.num_seeds)]
    else:
        runs = [(args.seed, fold) for fold in range(args.folds)]

    summaries = []
    for seed, fold in runs:
        summary = run_tu_graph(args, seed, fold, q_dev, w_shapes_dict, device)
        if summary is None:   # --count_params
            return
        summaries.append(summary)

    test_accs = [s['test_acc'] for s in summaries]
    mean_acc = float(np.mean(test_accs))
    std_acc = float(np.std(test_accs, ddof=1)) if len(test_accs) > 1 else None

    # One JSON per (model, dataset, config): everything that defines the experiment goes into the hash.
    not_config = ('plot', 'save_model', 'gradient', 'results', 'count_params', 'pre_train', 'continue_train',
                  'task', 'train_size', 'test_size', 'eval_size', 'fold', 'node_qubit')
    config = {k: v for k, v in sorted(vars(args).items()) if k not in not_config}
    if args.split_mode == 'holdout':
        config.pop('folds')
    else:
        config.pop('num_seeds')
        config.pop('test_frac')
    config_hash = hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]

    out = {
        'dataset': args.dataset.upper(),
        'model': args.model,
        'config': config,
        'protocol': {
            'split_mode': args.split_mode,
            'runs': 'one per seed' if args.split_mode == 'holdout' else 'one per fold',
            'selection': 'epoch with the best validation accuracy (ties: lower validation loss)',
            'reported_metric': 'test accuracy at the best-validation epoch',
            'early_stopping_patience': args.patience,
            'max_epochs': args.epochs,
        },
        'n_runs': len(summaries),
        'test_acc_mean': mean_acc,
        'test_acc_std': std_acc,
        'test_acc': test_accs,
        'val_acc': [s['val_acc'] for s in summaries],
        'best_epochs': [s['best_epoch'] for s in summaries],
        'seeds': [s['seed'] for s in summaries],
        'folds': [s['fold'] for s in summaries],
        'runs': summaries,
        'timestamp': timestamp,
    }
    summary_dir = os.path.join(result_dir, 'summary')
    os.makedirs(summary_dir, exist_ok=True)
    out_path = os.path.join(summary_dir, f"{args.dataset.lower()}_{args.model}_{args.split_mode}_{config_hash}.json")
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    std_text = f"{std_acc:.4f}" if std_acc is not None else "n/a"
    print(f"\n{args.model} on {args.dataset.upper()} | {args.split_mode} | {len(summaries)} runs")
    print(f"Test acc at best-validation epoch: {mean_acc:.4f} ± {std_text}")
    print(f"  per run:     {[round(a, 4) for a in test_accs]}")
    print(f"  best epochs: {out['best_epochs']}")
    print(f"Saved to {out_path}")


def main(args):
    set_seed(args.seed)
    if args.dataset.upper() in TU_GRAPH_DATASETS:
        main_tu(args)
    else:
        main_legacy(args)


if __name__ == "__main__":
    args = get_args()
    main(args)
