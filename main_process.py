import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric import nn
import os
import numpy as np
from torch_geometric.utils import degree
import pandas as pd
from torch_geometric.data import Data
# from torch_geometric.datasets import TUDataset
# from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Flickr
from collections import Counter
import time
import igraph as ig
import spectral_pro as sp

from add_mask import add_mask
from filter_top_features import filter_top_features
from reduce_features import reduce_features
from create_homo_graph import homomorphism_graph
from set_seed import set_seed
from images import plot_results

############################### set global seed for reproductivity
seed = 514
set_seed(514)
############################### get original dataset

# name = 'PubMed'
name = 'Cora'
# name = 'Citeseer'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists('./{}'.format(name)):
    os.makedirs('./{}'.format(name))

dataset = Planetoid(root='./{}'.format(name), name=name, split='public')
data = dataset[0]


############################### GCN model
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = nn.GCNConv(num_features, hidden_channels)
        self.conv2 = nn.GCNConv(hidden_channels, hidden_channels)
        self.conv3 = nn.GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)


# Train
def train(model,data,optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss

# Train with collecting features
def train_collect(model,data,optimizer,n):
    if not data.x.requires_grad:
        data.x.requires_grad = True
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    gradients = data.x.grad
    feature_importance = torch.sum(torch.abs(gradients), dim=0)
    top_n_indices = torch.topk(feature_importance, n).indices.cpu().numpy()
    optimizer.step()
    return loss, top_n_indices

# eval
def eval(model,data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
    acc_val = int(correct) / int(data.val_mask.sum())
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc_test = int(correct) / int(data.test_mask.sum())
    return acc_val, acc_test

################################ A clean model to record training time
def train_clean_model(dataset, device, data):
    model_clean = GCN(dataset.num_features, 128, dataset.num_classes).to(device)
    optimizer_clean = torch.optim.Adam(model_clean.parameters(), lr=0.01, weight_decay=5e-4)
    start_time = time.time()
    for epoch in range(100):
        loss = train(model_clean, data, optimizer_clean)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Total Training Time for Clean Model: {training_time:.4f} seconds")
    return training_time


############################### add train val test mask if not have

data = add_mask(data)


############################### train the original model get the important features
total_time = train_clean_model(dataset,device,data)
print(f"{name} dataset has {total_time:.4f} seconds training time.")
total_features = dataset.num_features
print(f"{name} dataset has {total_features} features.")
total_nodes = data.num_nodes
print(f"{name} dataset has {total_nodes} nodes.")
total_edges = data.num_edges
print(f"{name} dataset has {total_edges} edges.")

# top n important features by gradient
n = int(total_features/10)

important_features = []
modelCora = GCN(dataset.num_features, 128, dataset.num_classes).to(device)
optimizerCora = torch.optim.Adam(modelCora.parameters(), lr=0.01, weight_decay=5e-4)
for epoch in range(100):
    loss, top_features = train_collect(modelCora, data, optimizerCora, n)
    # top_features = get_top_n_features(modelCora, data, n, optimizerCora)
    important_features.extend(top_features)

acc_val, ori_acc_test = eval(modelCora, data)
print(f'Validation Accuracy for {name}: {acc_val:.4f}')
print(f'Test Accuracy for {name}: {ori_acc_test:.4f}')

feature_counts = Counter(important_features)
top_n_indices = [feature for feature, count in feature_counts.most_common(n)]
top_n_indices = [int(value) for value in top_n_indices]

print(f'Total number of important features collected: {len(important_features)}')
# print("Important features indices collected across epochs:", important_features)
# print("Top n most frequently important features:", top_n_indices)


########################################### get top m important categorical features
threshold = 10
m = int(n/2)

if m < 20:
    m = 20

m_test = [int(total_features*0.04),
          int(total_features*0.05), 
          int(total_features*0.06), 
          int(total_features*0.07), 
          int(total_features*0.08),
          int(total_features*0.09),
          int(total_features*0.10)]

results = {
    m: {
        'best_accuracy': {
            'value': 0,
            'time': 0,
            'edges': 0,
            'nodes': 0,
            'dist' : 0
        },
        'worst_accuracy': {
            'value': 10,
            'time': 0,
            'edges': 0,
            'nodes': 0,
            'dist' : 0
        },
        'fastest_time': {
            'value': float('inf'),
            'accuracy': 0,
            'edges': 0,
            'nodes': 0,
            'dist' : 0
        },
        'slowest_time': {
            'value': -float('inf'),
            'accuracy': 0,
            'edges': 0,
            'nodes': 0,
            'dist' : 0
        },
        'most_edges': {
            'value': 0,
            'accuracy': 0,
            'time': 0,
            'nodes': 0,
            'dist' : 0
        },
        'least_edges': {
            'value': float('inf'),
            'accuracy': 0,
            'time': 0,
            'nodes': 0,
            'dist' : 0
        },
        'most_nodes': {
            'value': 0,
            'accuracy': 0,
            'time': 0,
            'edges': 0,
            'dist' : 0
        },
        'least_nodes': {
            'value': float('inf'),
            'accuracy': 0,
            'time': 0,
            'edges': 0,
            'dist' : 0
        },
        'shortest_dist': {
            'value': float('inf'),
            'accuracy': 0,
            'time': 0,
            'edges': 0,
            'nodes' : 0,
            'eig_values': {}
        },
        'farthest_dist': {
            'value': -float('inf'),
            'accuracy': 0,
            'time': 0,
            'edges': 0,
            'nodes' : 0,
            'eig_values': {}
        }
    } for m in m_test
}


for j in range(0, len(m_test)):
    df_features_list = []
    top_m_categorical_features_list = []


    print("Current m: ", m_test[j])
    for k in range(0, 10):
        df_features, top_m_categorical_features = filter_top_features(data, top_n_indices, threshold, m_test[j])
        df_features_list.append(df_features)
        top_m_categorical_features_list.append(top_m_categorical_features)


    
    for i in range(0, len(df_features_list)):
        #################### create new dataset based on m categorical features

        df_features = df_features_list[i]
        top_m_categorical_features = top_m_categorical_features_list[i]

        df_new_features, data_new = reduce_features(df_features, top_m_categorical_features, data)

        data_star = homomorphism_graph(data_new, df_new_features, data)


        ############################# graph spectral eigenvalues

        dataset_names = ['ori', 'new']
        datasets = {}
        datasets["ori"] = data
        datasets["new"] = data_star
        igraph_graphs = {}
        for cur_name, cur_data in datasets.items():
            G = sp.pyg_data_to_igraph(cur_data)
            igraph_graphs[cur_name] = G

        current_dist, cur_eig_value = sp.spectral(dataset_names, igraph_graphs)
        
        ############################################### test and validation
        num_classes = data_star.y.unique().size(0)
        num_features = data_star.x.shape[1]
        modelCora_star = GCN(num_features, 128, num_classes).to(device)

        optimizerCora_star = torch.optim.Adam(modelCora_star.parameters(), lr=0.01, weight_decay=5e-4)

        start_homo = time.time()

        for epoch in range(100):
            loss = train(modelCora_star, data_star, optimizerCora_star)

        end_homo = time.time()
        homo_training_time = end_homo - start_homo
        # print(f"Total Training Time: {homo_training_time:.4f} seconds")

        # acc_val, acc_test = eval(modelCora_star, data_star)
        acc_val, acc_test = eval(modelCora_star, data_new)

        num_edges = data_star.edge_index.size(1)
        num_nodes = data_star.num_nodes
        # if acc_test > bestAcc:
        #     bestAcc = acc_test
        #     print(f'New best test accuracy: {bestAcc}')

        # if acc_test < worstAcc:
        #     worstAcc = acc_test
        #     print(f'New worst test accuracy: {worstAcc}')

        ################################################# Update records

        if acc_test > results[m_test[j]]['best_accuracy']['value']:
            results[m_test[j]]['best_accuracy'] = {
                'value': acc_test,
                'time': homo_training_time,
                'edges': num_edges,
                'nodes': num_nodes,
                'dist' : current_dist
            }

        if acc_test < results[m_test[j]]['worst_accuracy']['value']:
            results[m_test[j]]['worst_accuracy'] = {
                'value': acc_test,
                'time': homo_training_time,
                'edges': num_edges,
                'nodes': num_nodes,
                'dist' : current_dist
            }

        if homo_training_time < results[m_test[j]]['fastest_time']['value']:
            results[m_test[j]]['fastest_time'] = {
                'value': homo_training_time,
                'accuracy': acc_test,
                'edges': num_edges,
                'nodes': num_nodes,
                'dist' : current_dist
            }

        if homo_training_time > results[m_test[j]]['slowest_time']['value']:
            results[m_test[j]]['slowest_time'] = {
                'value': homo_training_time,
                'accuracy': acc_test,
                'edges': num_edges,
                'nodes': num_nodes,
                'dist' : current_dist
            }

        if num_edges > results[m_test[j]]['most_edges']['value']:
            results[m_test[j]]['most_edges'] = {
                'value': num_edges,
                'accuracy': acc_test,
                'time': homo_training_time,
                'nodes': num_nodes,
                'dist' : current_dist
            }

        if num_edges < results[m_test[j]]['least_edges']['value']:
            results[m_test[j]]['least_edges'] = {
                'value': num_edges,
                'accuracy': acc_test,
                'time': homo_training_time,
                'nodes': num_nodes,
                'dist' : current_dist
            }

        if num_nodes > results[m_test[j]]['most_nodes']['value']:
            results[m_test[j]]['most_nodes'] = {
                'value': num_nodes,
                'accuracy': acc_test,
                'time': homo_training_time,
                'edges': num_edges,
                'dist' : current_dist
            }

        if num_nodes < results[m_test[j]]['least_nodes']['value']:
            results[m_test[j]]['least_nodes'] = {
                'value': num_nodes,
                'accuracy': acc_test,
                'time': homo_training_time,
                'edges': num_edges,
                'dist' : current_dist
            }

        if current_dist < results[m_test[j]]['shortest_dist']['value']:
            results[m_test[j]]['shortest_dist'] = {
                'value': current_dist,
                'accuracy': acc_test,
                'time': homo_training_time,
                'edges': num_edges,
                'nodes' : num_nodes,
                'eig_values': cur_eig_value
            }

        if current_dist > results[m_test[j]]['farthest_dist']['value']:
            results[m_test[j]]['farthest_dist'] = {
                'value': current_dist,
                'accuracy': acc_test,
                'time': homo_training_time,
                'edges': num_edges,
                'nodes' : num_nodes,
                'eig_values': cur_eig_value
            }

        # print(f'Validation Accuracy for {name}_star: {acc_val:.4f}')
        # print(f'Test Accuracy for {name}_star: {acc_test:.4f}')
    print(f"Results for m = {m_test[j]}:")
    print(f"  Best Accuracy: {results[m_test[j]]['best_accuracy']}")
    print(f"  Worst Accuracy: {results[m_test[j]]['worst_accuracy']}")
    print(f"  Fastest Training Time: {results[m_test[j]]['fastest_time']}")
    print(f"  Slowest Training Time: {results[m_test[j]]['slowest_time']}")
    print(f"  Most Edges: {results[m_test[j]]['most_edges']}")
    print(f"  Least Edges: {results[m_test[j]]['least_edges']}")
    print(f"  Most Nodes: {results[m_test[j]]['most_nodes']}")
    print(f"  Least Nodes: {results[m_test[j]]['least_nodes']}")
    print(f"  Shortest Normalized Distance: {results[m_test[j]]['shortest_dist']['value']}")
    print(f"  Longest Normalized Distance: {results[m_test[j]]['farthest_dist']['value']}")


#################### draw plots
plot_results(m_test, total_features, results, ori_acc_test, total_time, total_edges, total_nodes, name)
sp.plot_spectral(results, m_test, total_features, name)