import torch
import random
from collections import defaultdict
# from torchmetrics.classification import MulticlassF1Score


def star_subgraph(adjacency_matrix, subgraph_size=4):
    num_nodes = adjacency_matrix.shape[0]
    subgraph_indices = []
    uncovered_neighbors = set(range(num_nodes))  # All nodes should be covered as neighbors at least once

    leaf_counts = defaultdict(int)

    seed_nodes = list(range(num_nodes))
    random.shuffle(seed_nodes)

    for center_node in seed_nodes:
        neighbors = [i for i in range(num_nodes) if adjacency_matrix[center_node, i] != 0 and i != center_node]
        k = subgraph_size - 1

        candidates = neighbors  # Already excludes center node

        # Case 1: Not enough neighbors → take all of them
        if len(candidates) <= k:
            sampled_neighbors = candidates

        else:
            available_new = list(set(candidates) & uncovered_neighbors)

            # Case 2a: enough new nodes → sample from them
            if len(available_new) >= k:
                sampled_neighbors = random.sample(available_new, k)

            # Case 2b: not enough new nodes → take all + fill from candidates
            else:
                sampled_neighbors = available_new
                remaining_k = k - len(sampled_neighbors)
                remaining_pool = list(set(candidates) - set(sampled_neighbors))
                remaining_pool.sort(key=lambda x: leaf_counts[x])

                sampled_neighbors += remaining_pool[:remaining_k]

        # Update uncovered neighbor set
        uncovered_neighbors -= set(sampled_neighbors)
        for node in sampled_neighbors:
            leaf_counts[node] += 1

        # Add center + its sampled neighbors
        subgraph = [center_node] + sampled_neighbors
        subgraph_indices.append(subgraph)

    return subgraph_indices


def train_graph(model, optimizer, loader, criterion, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_attr, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss)  * data.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def test_graph(model, loader, criterion, device, num_classes=0):
    model.eval()
    total_loss = 0
    correct = 0
    
    all_preds = []
    all_labels = []
    f1 = 0
    
    
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_attr, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        total_loss += float(loss) * data.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        
        all_preds.append(pred)
        all_labels.append(data.y)
    # if num_classes:   
    #     all_preds = torch.cat(all_preds, dim=0)
    #     all_labels = torch.cat(all_labels, dim=0)
    #     f1_metric = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)
    #     f1 = f1_metric(all_preds, all_labels)
    acc = correct / len(loader.dataset)
    return total_loss / len(loader.dataset), acc, f1


def train_node(model, optimizer, data, criterion, device, het_node_type=None):
    model.train()
    data = data.to(device)
    optimizer.zero_grad()
    if het_node_type is None:
        out = model(data.x, data.edge_attr, data.edge_index, batch=None)  # batch is unused
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
    else:
        out = model(data.x_dict, None, data.edge_index_dict)
        target = data[het_node_type]
        loss = criterion(out[target.train_mask], target.y[target.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test_node(model, data, criterion, device, het_node_type=None):
    model.eval()
    data = data.to(device)
    if het_node_type is None:
        out = model(data.x, data.edge_attr, data.edge_index, batch=None)
        target = data
    else:
        out = model(data.x_dict, None, data.edge_index_dict)
        target = data[het_node_type]

    results = {}
    for split in ['train', 'val', 'test']:
        mask = target[f'{split}_mask']
        loss = criterion(out[mask], target.y[mask])
        pred = out[mask].argmax(dim=1)
        correct = (pred == target.y[mask]).sum().item()
        acc = correct / mask.sum().item()

        results[split] = {
            'loss': float(loss),
            'acc': acc,
        }

    return results

def save_checkpoint(model, optimizer, save_path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, save_path)

