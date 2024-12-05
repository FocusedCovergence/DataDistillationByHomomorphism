import numpy as np
import igraph as ig
import matplotlib.pyplot as plt


def pyg_data_to_igraph(data):

    edge_index = data.edge_index.cpu().numpy()
    edges = list(zip(edge_index[0], edge_index[1]))
    G = ig.Graph(edges=edges, directed=True)

    return G

def spectral(dataset_names, igraph_graphs):
    eigenvalues = {}

    for name in dataset_names:
        G = igraph_graphs[name]
        adjacency = np.array(G.get_adjacency().data)
        eigenvalues[name] = np.linalg.eigvals(adjacency)

        # keep only the real part
        eigenvalues[name] = np.real(eigenvalues[name])

        ev = eigenvalues[name]
        eigenvalues[name] = ev / np.linalg.norm(ev)
        
    dist = 0
    for i in range(0, len(dataset_names)-1):
        name_i = dataset_names[i]
        name_j = dataset_names[i + 1]

        min_length = min(len(eigenvalues[name_i]), len(eigenvalues[name_j]))
        ev_i = np.sort(eigenvalues[name_i])[-min_length:]
        ev_j = np.sort(eigenvalues[name_j])[-min_length:]

        # compute distance
        dist = np.linalg.norm(ev_i - ev_j)
        print(f"Eigenvalue spectra distance between {name_i} and {name_j}: {dist}")

    return dist, eigenvalues

    # for name in dataset_names:
    #     ev = eigenvalues[name]
    #     eigenvalues[name] = ev / np.linalg.norm(ev)
    #     print(f"Normalized Eigenvalue spectra distance between {name_i} and {name_j}: {dist}")

def plot_spectral(results, m_test, total_features, name):

    m_ratios = [m / total_features for m in m_test]

    for i, m_ratio in enumerate(m_ratios):

        shortest_eig = results[m_test[i]]['shortest_dist']['eig_values']
        farthest_eig = results[m_test[i]]['farthest_dist']['eig_values']
        ev_short = np.sort(shortest_eig['new']) if 'new' in shortest_eig else []
        ev_far = np.sort(farthest_eig['new']) if 'new' in farthest_eig else []
        ev_ori = np.sort(farthest_eig['ori']) if 'ori' in farthest_eig else []

        plt.figure(figsize=(10, 6))
        plt.plot(ev_short, label='min')
        plt.plot(ev_far, label='max')
        plt.plot(ev_ori, label='ori')
        plt.xlabel('Eigenvalue Index')
        plt.ylabel('Normalized Eigenvalue')
        plt.title(f'Norm Eigenvalue Spectra Comparison (m_ratio={m_ratio:.2f} * total features)')
        plt.legend()
        plt.savefig(f'./images/distance_{name}_m_{m_ratio:.2f}.png')
        plt.close()
