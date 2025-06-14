import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import MLP, global_add_pool, global_mean_pool, global_max_pool   

from utils import star_subgraph


def message_passing_pqc(strong, twodesign, inits, wires):
    edge, center, neighbor, ancilla = wires
    ##
    # qml.StronglyEntanglingLayers(weights=strong[0], wires=[edge, center, neighbor])
    ##
    # qml.CRX(phi=inits[0,0],wires=[center, edge])
    # qml.CRX(phi=inits[0,1],wires=[center, neighbor])
    # qml.StronglyEntanglingLayers(weights=strong[0], wires=[edge, ancilla])
    # qml.StronglyEntanglingLayers(weights=strong[0], wires=[neighbor, ancilla])
    # qml.StronglyEntanglingLayers(weights=strong[0], wires=[center, ancilla])
    # qml.StronglyEntanglingLayers(weights=strong[1], wires=[edge, center])
    # qml.StronglyEntanglingLayers(weights=strong[2], wires=[center, neighbor])
    
    ## only rotation
    qml.CRX(inits[0, 0], wires=[neighbor, ancilla])
    qml.CRY(inits[0, 1], wires=[edge, ancilla])
    qml.CRZ(inits[0, 2], wires=[neighbor, ancilla])
    # qml.StronglyEntanglingLayers(weights=strong[0], wires=[edge, neighbor, ancilla])


def qgcn_enhance_layer(inputs, spreadlayer, strong, twodesign, inits, update):
    edge_feat_dim = feat_dim = node_feat_dim = 2
    inputs = inputs.reshape(-1,feat_dim)
    
    # The number of avaible nodes and edges
    total_shape = inputs.shape[0]
    num_nodes = (total_shape+1)//2
    num_edges = num_nodes - 1
    
    adjacency_matrix, vertex_features = inputs[:num_edges,:], inputs[num_edges:,:]

    # The number of qubits assiged to each node and edge
    num_qbit = spreadlayer.shape[1]
    num_nodes_qbit = (num_qbit+1)//2
    num_edges_qbit = num_nodes_qbit - 1
    
    center_wire = num_edges_qbit
    
    
    for i in range(num_edges):
        qml.RY(adjacency_matrix[i][0], wires=i)
        qml.RZ(adjacency_matrix[i][1], wires=i)
        # qml.RX(adjacency_matrix[i][2], wires=i)
    
    for i in range(num_nodes):
        qml.RY(vertex_features[i][0], wires=center_wire+i)
        qml.RZ(vertex_features[i][1], wires=center_wire+i)
        # qml.RX(vertex_features[i][2], wires=center_wire+i)
    
    
    for i in range(num_edges):

        message_passing_pqc(strong=strong, twodesign=twodesign, inits=inits, 
                            wires=[i, center_wire, center_wire+i+1, num_qbit])

    qml.StronglyEntanglingLayers(
        weights=update[0], 
        wires=[center_wire, num_qbit]
        )
    probs = qml.probs(wires=[center_wire, num_qbit])
    return probs


def small_normal_init(tensor):
    return torch.nn.init.normal_(tensor, mean=0.0, std=0.1)


class QGNNGraphClassifier(nn.Module):
    def __init__(self, q_dev, w_shapes, node_input_dim=1, edge_input_dim=1,
                 graphlet_size=4, hop_neighbor=1, num_classes=2, one_hot=0):
        super().__init__()
        self.hidden_dim = 128
        self.graphlet_size = graphlet_size
        self.one_hot = one_hot
        self.hop_neighbor = hop_neighbor
        self.pqc_dim = 2 # number of feat per pqc for each node
        self.chunk = 1
        self.final_dim = self.pqc_dim * self.chunk # 2
        self.pqc_out = 4 # probs?
        
        self.qconvs = nn.ModuleDict()
        self.upds = nn.ModuleDict()
        self.aggs = nn.ModuleDict()
        self.norms = nn.ModuleDict()
        
        if self.one_hot:
            self.node_input_dim = 1
            self.edge_input_dim = 1
        else:
            self.node_input_dim = node_input_dim
            self.edge_input_dim = edge_input_dim if edge_input_dim > 0 else 1

        
        self.input_node = MLP(
                    [self.node_input_dim, self.hidden_dim, self.final_dim],
                    act='leaky_relu', 
                    norm=None, dropout=0.2
            )

        self.input_edge = MLP(
                    [self.edge_input_dim, self.hidden_dim, self.pqc_dim],
                    act='leaky_relu', 
                    norm=None, dropout=0.2
            )
        
        for i in range(self.hop_neighbor):
            qnode = qml.QNode(qgcn_enhance_layer, q_dev,  interface="torch")
            self.qconvs[f"lay{i+1}"] = qml.qnn.TorchLayer(qnode, w_shapes, small_normal_init)
            
            self.upds[f"lay{i+1}"] = MLP(
                    [self.pqc_dim + self.pqc_out, self.hidden_dim, self.pqc_dim],
                    act='leaky_relu', 
                    norm=None, dropout=0.2
            )
            
            self.norms[f"lay{i+1}"] = nn.LayerNorm(self.pqc_dim)
            
        self.graph_head = MLP(
                [self.final_dim, num_classes, num_classes],
                act='leaky_relu', 
                norm=None, dropout=0.5
        ) 
        
    def forward(self, node_feat, edge_attr, edge_index, batch):
        edge_index = edge_index.t()
        num_nodes = node_feat.size(0)
        num_nodes_model = self.graphlet_size
        num_edges_model = self.graphlet_size - 1
        
        if edge_attr is None:
            edge_attr = torch.ones((edge_index.size(0), self.edge_input_dim), device=node_feat.device)
        
        edge_features = edge_attr.float()
        node_features = node_feat.float()
        
        edge_features = self.input_edge(edge_features)
        node_features = self.input_node(node_features)
        
        edge_features = torch.tanh(edge_features) * np.pi
        node_features = torch.tanh(node_features) * np.pi
        
        idx_dict = {
            (int(u), int(v)): i
            for i, (u, v) in enumerate(edge_index.tolist())
        }
        

        adj_mtx = torch.zeros((num_nodes, num_nodes), dtype=torch.int)
        adj_mtx[edge_index[:, 0], edge_index[:, 1]] = 1
        adj_mtx[edge_index[:, 1], edge_index[:, 0]] = 1
        
        
        for i in range(self.hop_neighbor):
            subgraphs = star_subgraph(adj_mtx.cpu().numpy(), subgraph_size=self.graphlet_size)
            node_upd = torch.zeros((num_nodes, self.final_dim), device=node_features.device)
            q_layer = self.qconvs[f"lay{i+1}"]
            upd_layer = self.upds[f"lay{i+1}"]
            norm_layer = self.norms[f"lay{i+1}"]
            # agg_layer = self.aggs[f"lay{i+1}"]
            
            # updates = [[] for _ in range(num_nodes)]
            updates_node = node_features.clone() ## UPDATES
            
            ## TODO: Trying chunking PQCs  
            for sub in subgraphs:
                center = sub[0]
                neighbors = sub[1:]

                n_feat = node_features[sub]  # shape: [len(sub), d_n]
                
                edge_idxs = [idx_dict[(center, int(n))] for n in neighbors]
                e_feat = edge_features[edge_idxs]  # shape: [len(neighbors), d_e]
                n_feat = n_feat.reshape(len(sub),self.pqc_dim,-1)   
                
                for i in range(n_feat.shape[2]):
                    inputs = torch.cat([e_feat, n_feat[:,:,i]], dim=0)   
                    all_msg = q_layer(inputs.flatten()).reshape(-1,2)
                    # aggr = torch.sum(all_msg, dim=0)
                    
                    aggr = all_msg.flatten()
                    update_vec  = upd_layer(torch.cat([node_features[center, i*self.pqc_dim:(i+1)*self.pqc_dim], aggr], dim=0))
                    # update_vec = aggr
                    update_vec = torch.tanh(update_vec) * np.pi
                    updates_node[center, i*self.pqc_dim:(i+1)*self.pqc_dim] = (updates_node[center, i*self.pqc_dim:(i+1)*self.pqc_dim] + update_vec)/2 
                    
            ## TODO: End test section 
            # for sub in subgraphs:
            #     center = sub[0]
            #     neighbors = sub[1:]

            #     n_feat = node_features[sub] 
            #     # e_feat = edge_attributes[center, neighbors] 
            #     edge_idxs = [ idx_dict[(center, int(n))] for n in neighbors ]
            #     e_feat    = edge_features[edge_idxs]  
                
            #     inputs = torch.cat([e_feat, n_feat], dim=0)        

            #     all_msg = q_layer(inputs.flatten()).reshape(-1,2)
            #     aggr = torch.sum(all_msg, dim=0)
                
            #     update_vec  = upd_layer(torch.cat([node_features[center], aggr], dim=0))
            #     # updates[center].append(new_center)
            #     updates_node[center] += update_vec  
            ## TODO: End original section 
            # updates_node = F.relu(updates_node)
            node_features = norm_layer(updates_node + node_features)    
            # updates_node = []
            # for update in updates:
            #     updates_node.append(torch.stack(update))
                
            # node_features = F.relu(torch.vstack(updates_node))
        # node_features = self.final(node_features)
        graph_embedding = global_mean_pool(node_features, batch)
        
        return self.graph_head(graph_embedding)
