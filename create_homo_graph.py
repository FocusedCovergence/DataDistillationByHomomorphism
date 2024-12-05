import torch
from torch_geometric.data import Data
import numpy as np

def homomorphism_graph(data_new, df_new_features, data):

    features_new = data_new.x.numpy()
    num_nodes = features_new.shape[0]

    # clusters as new nodes
    grouped_nodes = {}

    for node_idx in range(num_nodes):
        feature_tuple = tuple(df_new_features.iloc[node_idx].values)
        
        # group nodes by feature combination
        if feature_tuple not in grouped_nodes:
            grouped_nodes[feature_tuple] = []
        
        grouped_nodes[feature_tuple].append(node_idx)

    new_nodes = list(grouped_nodes.values())
    num_new_nodes = len(new_nodes)

    # Create a mapping from original nodes to new group indices
    original_to_group = {}
    for new_node_idx, group in enumerate(new_nodes):
        for original_node in group:
            original_to_group[original_node] = new_node_idx


    # get original edge index
    original_edge_index = data_new.edge_index
    
    distilled_edge_set = set()

    # Iterate over original edges to construct edges for distilled graph
    for i in range(original_edge_index.shape[1]):
        source = original_edge_index[0, i].item()
        target = original_edge_index[1, i].item()

        source_group = original_to_group[source]
        target_group = original_to_group[target]

        # different group add edge
        # same group add self loop
        if source_group != target_group:
            distilled_edge_set.add((source_group, target_group))
        else:
            distilled_edge_set.add((source_group, source_group))

    distilled_edge_index = torch.tensor(list(distilled_edge_set), dtype=torch.long).t()


    new_node_features = []

    for group in new_nodes:
        new_node_features.append(df_new_features.iloc[group[0]].values)

    new_node_features = np.array(new_node_features)
    new_node_features = torch.tensor(new_node_features, dtype=torch.float)

    original_labels = data.y.numpy()
    new_node_labels = []

    for group in new_nodes:
        group_labels = original_labels[group]

        # mode for new node label
        new_label = np.bincount(group_labels).argmax()
        new_node_labels.append(new_label)

    new_node_labels = torch.tensor(new_node_labels, dtype=torch.long)

    # mask
    train_mask = torch.zeros(num_new_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_new_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_new_nodes, dtype=torch.bool)

    num_train = int(0.8 * num_new_nodes)
    num_val = int(0.1 * num_new_nodes)
    num_test = int(0.1 * num_new_nodes)

    indices = torch.randperm(num_new_nodes)

    train_mask[indices[:num_train]] = True
    val_mask[indices[num_train:num_train + num_val]] = True
    test_mask[indices[num_train + num_val:num_train + num_val + num_test]] = True

    data_star = Data(x=new_node_features, edge_index=distilled_edge_index, y=new_node_labels,
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    # print(data_star)
    # num_classes = data_star.y.unique().size(0)
    # print(f"Number of classes: {num_classes}")
    return data_star