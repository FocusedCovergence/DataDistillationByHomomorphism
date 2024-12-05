import matplotlib.pyplot as plt
import os
def plot_results(m_test, total_features, results, ori_acc_test, normal_time, total_edges, total_nodes, name):
    if not os.path.exists('./images'):
        os.makedirs('./images')

    m_ratios = [m / total_features for m in m_test]

    best_acc = [results[m]['best_accuracy']['value'] for m in m_test]
    worst_acc = [results[m]['worst_accuracy']['value'] for m in m_test]
    fastest_time = [results[m]['fastest_time']['value'] for m in m_test]
    slowest_time = [results[m]['slowest_time']['value'] for m in m_test]
    most_edges = [results[m]['most_edges']['value'] for m in m_test]
    least_edges = [results[m]['least_edges']['value'] for m in m_test]
    most_nodes = [results[m]['most_nodes']['value'] for m in m_test]
    least_nodes = [results[m]['least_nodes']['value'] for m in m_test]

    # accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(m_ratios, best_acc, label='Best Accuracy', marker='o')
    plt.plot(m_ratios, worst_acc, label='Worst Accuracy', marker='o')
    plt.axhline(y=ori_acc_test, color='r', linestyle='--', label='Original Accuracy')
    plt.xlabel('m (ratio to total features)')
    plt.ylabel('Accuracy')
    plt.title(f'{name} set: Accuracy vs. Ratio of Total Features')
    plt.legend()
    plt.savefig(f'./images/accuracy_{name}.png')
    plt.close()

    # training time
    plt.figure(figsize=(10, 6))
    plt.plot(m_ratios, fastest_time, label='Fastest Training Time', marker='o')
    plt.plot(m_ratios, slowest_time, label='Slowest Training Time', marker='o')
    plt.axhline(y=normal_time, color='r', linestyle='--', label='Original Training Time')
    plt.xlabel('m (ratio to total features)')
    plt.ylabel('Training Time (seconds)')
    plt.title(f'{name} set: Training Time vs. Ratio of Total Features')
    plt.legend()
    plt.savefig(f'./images/time_{name}.png')
    plt.close()

    # edges
    plt.figure(figsize=(10, 6))
    plt.plot(m_ratios, most_edges, label='Most Edges', marker='o')
    plt.plot(m_ratios, least_edges, label='Least Edges', marker='o')
    plt.axhline(y=total_edges, color='r', linestyle='--', label='Total Edges (Original Graph)')
    plt.xlabel('m (ratio to total features)')
    plt.ylabel('Number of Edges')
    plt.title(f'{name} set: Edges vs. Ratio of Total Features')
    plt.legend()
    plt.savefig(f'./images/edges_{name}.png')
    plt.close()

    # nodes
    plt.figure(figsize=(10, 6))
    plt.plot(m_ratios, most_nodes, label='Most Nodes', marker='o')
    plt.plot(m_ratios, least_nodes, label='Least Nodes', marker='o')
    plt.axhline(y=total_nodes, color='r', linestyle='--', label='Total Nodes (Original Graph)')
    plt.xlabel('m (ratio to total features)')
    plt.ylabel('Number of Nodes')
    plt.title(f'{name} set: Nodes vs. Ratio of Total Features')
    plt.legend()
    plt.savefig(f'./images/nodes_{name}.png')
    plt.close()