"""
Graph Neural Network model for fraud detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.data import Data
import numpy as np

class FraudDetectionGNN(nn.Module):
    """
    Graph Neural Network for fraud detection
    Supports multiple architectures: GCN, GraphSAGE, GAT
    """
    def __init__(self, num_features, hidden_channels=64, num_classes=2, 
                 architecture='GCN', dropout=0.5):
        super(FraudDetectionGNN, self).__init__()
        
        self.architecture = architecture
        self.dropout = dropout
        
        if architecture == 'GCN':
            self.conv1 = GCNConv(num_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.conv3 = GCNConv(hidden_channels, num_classes)
            
        elif architecture == 'GraphSAGE':
            self.conv1 = SAGEConv(num_features, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
            self.conv3 = SAGEConv(hidden_channels, num_classes)
            
        elif architecture == 'GAT':
            self.conv1 = GATConv(num_features, hidden_channels, heads=4, dropout=dropout)
            self.conv2 = GATConv(hidden_channels*4, hidden_channels, heads=4, dropout=dropout)
            self.conv3 = GATConv(hidden_channels*4, num_classes, heads=1, dropout=dropout)
        
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
    
    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass
        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Graph connectivity [2, num_edges]
            edge_weight: Edge weights [num_edges] (optional)
        """
        # Layer 1
        if self.architecture == 'GCN':
            x = self.conv1(x, edge_index, edge_weight)
        else:
            x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 2
        if self.architecture == 'GCN':
            x = self.conv2(x, edge_index, edge_weight)
        else:
            x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 3 (output)
        if self.architecture == 'GCN':
            x = self.conv3(x, edge_index, edge_weight)
        else:
            x = self.conv3(x, edge_index)
        
        return F.log_softmax(x, dim=1)

class GNNTrainer:
    """
    Trainer for GNN model
    """
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = None
        self.criterion = None
        
    def setup_training(self, learning_rate=0.01, weight_decay=5e-4, class_weights=None):
        """Setup optimizer and loss function"""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Handle class imbalance
        if class_weights is not None:
            class_weights = torch.FloatTensor(class_weights).to(self.device)
        
        self.criterion = nn.NLLLoss(weight=class_weights)
    
    def train_epoch(self, data):
        """Train for one epoch"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        out = self.model(data.x, data.edge_index, data.edge_attr)
        
        # Compute loss only on training nodes
        loss = self.criterion(out[data.train_mask], data.y[data.train_mask])
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, data, mask):
        """Evaluate model"""
        self.model.eval()
        
        with torch.no_grad():
            out = self.model(data.x, data.edge_index, data.edge_attr)
            pred = out.argmax(dim=1)
            
            correct = (pred[mask] == data.y[mask]).sum()
            acc = int(correct) / int(mask.sum())
            
            # Get probabilities for fraud class
            probs = torch.exp(out[:, 1])
        
        return acc, pred, probs
    
    def train(self, data, epochs=200, verbose=True):
        """Full training loop"""
        best_val_acc = 0
        history = {'train_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            loss = self.train_epoch(data)
            
            if epoch % 10 == 0:
                val_acc, _, _ = self.evaluate(data, data.val_mask)
                history['train_loss'].append(loss)
                history['val_acc'].append(val_acc)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                
                if verbose and epoch % 20 == 0:
                    print(f'Epoch {epoch:03d}: Loss={loss:.4f}, Val Acc={val_acc:.4f}')
        
        if verbose:
            print(f'Best validation accuracy: {best_val_acc:.4f}')
        
        return history

def prepare_graph_data(G, node_features, node_to_idx, labels=None, 
                       train_ratio=0.6, val_ratio=0.2):
    """
    Convert NetworkX graph to PyTorch Geometric Data object
    
    Args:
        G: NetworkX graph
        node_features: Dict of node features
        node_to_idx: Mapping from node names to indices
        labels: Optional labels for nodes
        train_ratio, val_ratio: Split ratios
    
    Returns:
        PyTorch Geometric Data object
    """
    # Create edge index
    edge_list = []
    edge_weights = []
    
    for u, v, data in G.edges(data=True):
        u_idx = node_to_idx[u]
        v_idx = node_to_idx[v]
        edge_list.append([u_idx, v_idx])
        edge_weights.append(data.get('weight', 1.0))
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_weights, dtype=torch.float)
    
    # Normalize edge weights
    edge_attr = edge_attr / edge_attr.max()
    
    # Create node feature matrix
    feature_keys = sorted(list(node_features[list(node_features.keys())[0]].keys()))
    num_nodes = len(node_to_idx)
    num_features = len(feature_keys)
    
    x = torch.zeros((num_nodes, num_features), dtype=torch.float)
    
    for node, idx in node_to_idx.items():
        features = node_features[node]
        x[idx] = torch.tensor([features[key] for key in feature_keys], dtype=torch.float)
    
    # Normalize features
    x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)
    
    # Create masks and labels
    num_nodes = x.size(0)
    
    if labels is not None:
        y = torch.zeros(num_nodes, dtype=torch.long)
        for node, label in labels.items():
            if node in node_to_idx:
                y[node_to_idx[node]] = label
        
        # Create train/val/test split
        indices = torch.randperm(num_nodes)
        train_size = int(train_ratio * num_nodes)
        val_size = int(val_ratio * num_nodes)
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size+val_size]] = True
        test_mask[indices[train_size+val_size:]] = True
    else:
        y = None
        train_mask = torch.ones(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    # Create Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )
    
    return data
