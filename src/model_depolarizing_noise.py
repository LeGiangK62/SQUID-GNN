import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
from torch_geometric.nn import global_add_pool

# --- 1. CÁC HÀM MẠCH QUANTUM (PQC) ---

def apply_noise(noise_prob, wires):
    """Hàm phụ trợ để chèn nhiễu vào danh sách dây chỉ định"""
    if noise_prob > 0:
        for w in wires:
            qml.DepolarizingChannel(noise_prob, wires=w)

def entangle_circuit(strong, inits, wires, noise_prob):
    """Mạch đan xen với nhiễu sau mỗi tương tác 2-qubit"""
    w0, _, w1 = wires
    num_ent_layer = strong.shape[0]
    for i in range(num_ent_layer):
        qml.RY(inits[i, 1], wires=w0)
        qml.RY(inits[i, 2], wires=w1)
        
        # Tương tác 1
        qml.CRX(strong[i, 0], wires=[w0, w1])
        qml.CRX(strong[i, 1], wires=[w1, w0])
        apply_noise(noise_prob, [w0, w1]) # Nhiễu sau tương tác 2-qubit
        
        qml.RY(inits[i, 2], wires=w0)
        qml.RY(inits[i, 3], wires=w1)
        
        # Tương tác 2
        qml.CRX(strong[i, 2], wires=[w0, w1])
        qml.CRX(strong[i, 3], wires=[w1, w0])
        apply_noise(noise_prob, [w0, w1]) # Nhiễu sau tương tác 2-qubit

def qgcn_enhance_layer_core(inputs, strong, inits, update, noise_prob):
    """Hàm lõi với nhiễu xuất hiện xuyên suốt các giai đoạn"""
    feat_dim = 2
    inputs = inputs.reshape(-1, feat_dim)
    num_nodes = (inputs.shape[0] + 1) // 2
    num_edges = num_nodes - 1
    
    edge_start = 0
    node_start = num_edges 
    aux_start = num_edges + num_nodes 
    center_wire = node_start 

    # 1. Encoding + Noise (Lỗi khởi tạo trạng thái)
    for i in range(num_edges):
        qml.RY(inputs[i][0], wires=edge_start + i)
        qml.RZ(inputs[i][1], wires=edge_start + i)
    for i in range(num_nodes):
        qml.RY(inputs[num_edges + i][0], wires=node_start + i)
        qml.RZ(inputs[num_edges + i][1], wires=node_start + i)
    
    # Nhiễu sau khi encoding cho toàn bộ qubit đang hoạt động
    apply_noise(noise_prob, range(aux_start))

    # 2. Entanglement (Giai đoạn truyền tin - nhiễu tích hợp bên trong)
    for i in range(num_edges):
        entangle_circuit(strong, inits, wires=[i, center_wire, node_start + i + 1], noise_prob=noise_prob)

    # 3. Update Layer (Giai đoạn tính toán)
    for i in range(num_edges):
        u_wires = [center_wire, node_start + i + 1, aux_start, aux_start + 1]
        qml.StronglyEntanglingLayers(weights=update[i], wires=u_wires)
        apply_noise(noise_prob, u_wires) # Nhiễu sau mỗi block Update

    # 4. CHÈN NOISE CUỐI CÙNG (Lỗi đo lường)
    apply_noise(noise_prob, range(aux_start + 2))

    return [
        qml.expval(qml.PauliZ(center_wire)),
        qml.expval(qml.PauliZ(aux_start)),
        qml.expval(qml.PauliZ(aux_start + 1)),
    ]

# --- 2. LỚP MLP VÀ MODEL CHÍNH (Giữ nguyên cấu trúc Signature Wrapper) ---

class MLP(nn.Module):
    def __init__(self, dims, act='leaky_relu', norm=None, dropout=0.0):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if norm == 'batch_norm': layers.append(nn.BatchNorm1d(dims[i+1]))
            if act == 'leaky_relu': layers.append(nn.LeakyReLU())
            if dropout > 0: layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    def forward(self, x): return self.model(x)

class QGNNGraphClassifier(nn.Module):
    def __init__(self, q_dev, w_shapes, hidden_dim, node_input_dim, edge_input_dim,
                 graphlet_size=4, hop_neighbor=1, num_classes=2, noise_prob=0.0):
        super().__init__()
        self.q_dev = q_dev
        self.graphlet_size = graphlet_size
        self.hop_neighbor = hop_neighbor
        self.pqc_dim, self.pqc_out, self.final_dim = 2, 3, 2

        self.input_node = MLP([node_input_dim, hidden_dim, self.final_dim], act='leaky_relu', norm='batch_norm')
        self.input_edge = MLP([edge_input_dim, hidden_dim, self.pqc_dim], act='leaky_relu', norm='batch_norm')
        
        self.qconvs, self.upds, self.norms = nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict()
        clean_w_shapes = {k: v for k, v in w_shapes.items() if k in ['strong', 'inits', 'update']}

        for i in range(self.hop_neighbor):
            curr_p = noise_prob
            def circuit_wrapper(inputs, strong, inits, update):
                return qgcn_enhance_layer_core(inputs, strong, inits, update, curr_p)

            qnode = qml.QNode(circuit_wrapper, self.q_dev, interface="torch")
            self.qconvs[f"lay{i+1}"] = qml.qnn.TorchLayer(qnode, clean_w_shapes, nn.init.uniform_)
            self.upds[f"lay{i+1}"] = MLP([self.pqc_dim + self.pqc_out, hidden_dim, self.pqc_dim], dropout=0.4)
            self.norms[f"lay{i+1}"] = nn.LayerNorm(self.pqc_dim)

        self.graph_head = MLP(
                [self.final_dim, hidden_dim, hidden_dim, num_classes],
                act='leaky_relu', norm='batch_norm', dropout=0.1
        ) 

    def forward(self, node_feat, edge_attr, edge_index, batch):
        edge_index = edge_index.t()
        node_feat = node_feat.float()
        if edge_attr is None or edge_attr.numel() == 0:
            edge_attr = torch.ones((edge_index.size(0), self.input_edge.model[0].in_features), device=node_feat.device)
        else:
            edge_attr = edge_attr.float()
            if edge_attr.ndim == 1: edge_attr = edge_attr.unsqueeze(1)

        node_features = torch.tanh(self.input_node(node_feat)) * np.pi
        edge_features = torch.tanh(self.input_edge(edge_attr)) * np.pi
        
        for i in range(self.hop_neighbor):
            q_layer, upd_layer, norm_layer = self.qconvs[f"lay{i+1}"], self.upds[f"lay{i+1}"], self.norms[f"lay{i+1}"]
            updates_node = torch.zeros_like(node_features)
            
            for center in torch.unique(edge_index[:, 1]):
                mask = (edge_index[:, 1] == center)
                neighbor_ids = edge_index[:, 0][mask][:self.graphlet_size-1]
                edge_ids = torch.nonzero(mask).view(-1)[:self.graphlet_size-1]
                
                q_input = torch.cat([edge_features[edge_ids], node_features[center].unsqueeze(0), node_features[neighbor_ids]], dim=0).flatten()
                all_msg = q_layer(q_input)
                updates_node[center] = upd_layer(torch.cat([node_features[center], all_msg], dim=0))
            
            node_features = norm_layer(updates_node) + node_features

        return self.graph_head(global_add_pool(node_features, batch))