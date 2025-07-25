import os
import torch
import matplotlib.pyplot as plt
import pennylane as qml
import argparse
from torch import nn, optim
import numpy as np


from utils import train_graph, test_graph, EarlyStopping, save_checkpoint
from data import load_dataset, eval_dataset, random_split
from model import QGNNGraphClassifier, QGNNNodeClassifier
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
    parser.add_argument('--train_size', type=int, default=100)
    parser.add_argument('--eval_size', type=int, default=150)
    parser.add_argument('--test_size', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-2)
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--node_qubit', type=int, default=3)
    parser.add_argument('--num_gnn_layers', type=int, default=2)
    parser.add_argument('--num_ent_layers', type=int, default=1)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--seed', type=int, default=1712)
    parser.add_argument('--task', type=str, default='graph', choices=['graph', 'node'], help='graph or node classification')

    
    # Debug options
    parser.add_argument('--pre_train', type=str, default=None, help='Load the pre-trained model (timestamp as name)')
    parser.add_argument('--continue_train', action='store_true', help='Continue training from pre-trained model')
    parser.add_argument('--plot', action='store_true', help='Enable plotting')
    parser.add_argument('--save_model', action='store_true', help='Enable saving model')
    parser.add_argument('--gradient', action='store_true', help='Enable gradient saving')
    parser.add_argument('--results', action='store_true', help='Evaluate results')
    
    # For switching between models
    parser.add_argument('--model', type=str, default='qgnn', 
                        choices=['qgnn', 'handcraft', 'gin', 'gcn', 'gat', 'sage', 'trans'],
                        help="Which model to run"
                        )
    parser.add_argument('--graphlet_size', type=int, default=10)
    
    
    return parser.parse_args()


def main(args):
    args.node_qubit = args.graphlet_size
    edge_qubit = args.node_qubit - 1
    n_qubits = args.node_qubit + edge_qubit
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_dev = qml.device("default.qubit", wires=n_qubits + 2) # number of ancilla qubits

    # PQC weight shape settings
    w_shapes_dict = {
        'spreadlayer': (0, n_qubits, 1),
        # Old
        # 'strong': (2, args.num_ent_layers, 3, 3), # 3
        # # 'strong': (3, args.num_ent_layers, 2, 3), # 2
        # 'inits': (1, 4),
        # 'update': (1, args.num_ent_layers, 3, 3), # (1, args.num_ent_layers, 2, 3)
        # NEW
        'inits': (1, 2), # New
        'strong': (1, args.num_ent_layers, 2, 3), # New
        'update': (args.graphlet_size, args.num_ent_layers-1, 4, 3),
        'twodesign': (0, args.num_ent_layers, 1, 2)
    }

    # Load dataset
    dataset, train_loader, test_loader, task_type = load_dataset(
        name=args.dataset,
        path='../data',
        train_size=args.train_size,
        test_size=args.test_size,
        batch_size=args.batch_size
    )
    
    result_base = f"{timestamp}_{args.model}_{args.graphlet_size}_{args.epochs}_{args.lr}"
    plot_train_path = os.path.join(result_dir, 'fig', f"{args.dataset.lower()}_plot_{result_base}_train.png")
    npz_path = os.path.join(result_dir, 'train_plot', f"{args.dataset.lower()}_data_{result_base}.npz")
    model_save = os.path.join(result_dir, 'model', f"{args.dataset.lower()}_model_{result_base}.pt")

    # if task_type != 'graph':
    #     raise NotImplementedError("Node classification support is not implemented yet.")
 
    # Model metadata
    node_input_dim = dataset[0].x.shape[1] if dataset[0].x is not None else 0
    edge_input_dim = dataset[0].edge_attr.shape[1] if dataset[0].edge_attr is not None else 0
    num_classes = dataset.num_classes
    # Model init
    if args.task == 'graph':
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
                one_hot=0
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
    elif args.task == 'node':
        data = dataset[0].to(device)
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
                one_hot=0
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
        elif args.model == 'gin':
            from baseline import GIN_Node
            model = GIN_Node(
                in_channels=node_input_dim,
                hidden_channels=args.hidden_channels,
                out_channels=num_classes,
                num_layers=args.num_gnn_layers,
            )
        elif args.model == 'gcn':
            from baseline import GCN_Node
            model = GCN_Node(
                in_channels=node_input_dim,
                hidden_channels=args.hidden_channels,
                out_channels=num_classes,
                num_layers=args.num_gnn_layers,
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
        else:
            raise ValueError(f"Unsupported model for node task: {args.model}")
    else:
        raise ValueError("Unsupported task type")
    
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    criterion = nn.CrossEntropyLoss()
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
        pre_trained_path = os.path.join(result_dir, 'model', f"model_{args.pre_train}.pt")
        checkpoint = torch.load(pre_trained_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        pre_train_npz_path = os.path.join(result_dir, 'train_plot', f"data_{args.pre_train}.npz")
        data = np.load(pre_train_npz_path)
        pre_train_epoch = data['epoch'].shape[0]          
        train_losses = data['train_losses'].tolist()
        test_losses = data['test_losses'].tolist()
        train_accs = data['train_accs'].tolist()
        test_accs = data['test_accs'].tolist()
        print(f"Pre-trained model loaded from {pre_trained_path} with {pre_train_epoch} epochs.")
        if not args.continue_train: 
            model.eval()
            print("Skip training...")
        else:
            print(f"Continuing training model with {args.graphlet_size} graphlet size with {args.epochs} epochs, "
            f"learning rate {args.lr}, step size {args.step_size}, and gamma {args.gamma}.")
    else: 
        print(f"Training model with {args.graphlet_size} graphlet size with {args.epochs} epochs, "
            f"learning rate {args.lr}, step size {args.step_size}, and gamma {args.gamma}.")
            
        
        
        
    print(f"Training model {args.model} on {args.dataset} with {args.graphlet_size} graphlet size with {args.epochs} epochs, "
          f"learning rate {args.lr}, step size {args.step_size}, and gamma {args.gamma}.")
    if args.continue_train or args.pre_train is None:
    
        if args.task == 'graph':
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
                    print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                        f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
        else:  # node task
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min',                           
                factor=args.gamma,      # Multiplies LR by this factor (e.g., 0.5)
                patience=args.epochs//10,# Wait this many epochs without improvement
                # verbose=True                            # Print updates
            )
            from utils import train_node, test_node
            for epoch in range(1, args.epochs + 1):
                train_loss = train_node(model, optimizer, data, criterion, device)
                test_metrics = test_node(model, data, criterion, device, num_classes)
                train_losses.append(test_metrics['train']['loss'])
                test_losses.append(test_metrics['test']['loss'])
                train_accs.append(test_metrics['train']['acc'])
                test_accs.append(test_metrics['test']['acc'])
                
                if args.save_model:
                    # early_stopping(test_losses[-1], model)
                    save_checkpoint(model, optimizer, model_save)
                                        
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
            eval_loader = eval_dataset(
                name=args.dataset,
                path='../data',
                eval_size=args.eval_size,
                batch_size=args.batch_size,
                seed=args.seed+each
            )
            if args.task == 'graph':
                _, eval_acc, _ = test_graph(model, eval_loader, criterion, device, num_classes)
            elif args.task == 'node':
                eval_loader = random_split(eval_loader, train_ratio=0.6, val_ratio=0.2, seed=args.seed+each)
                eval_metrics = test_node(model, eval_loader, criterion, device, num_classes)
                eval_acc = eval_metrics['val']['acc']
            else:
                raise ValueError(f"Unsupported task: {args.task}")
            
            accuracies.append(eval_acc)

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies, ddof=1)  # unbiased std deviation

        print(f"{args.model} Mean Accuracy: {mean_acc:.4f} ± {std_acc:.3f}")

if __name__ == "__main__":
    args = get_args()
    main(args)
