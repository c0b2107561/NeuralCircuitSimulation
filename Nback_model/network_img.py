import networkx as nx
import matplotlib.pyplot as plt

def Watts_Strogats_small_world_graph(n_nodes, k_neighbors, p_rewire):

    ws_graph = nx.watts_strogatz_graph(n_nodes, k_neighbors, p_rewire)
    return ws_graph

def graph_plot(graph):
    pos = nx.circular_layout(graph)
    # nx.draw(graph, pos, with_labels=True, font_weight='bold')
    # plt.show()

    nx.draw(graph, with_labels=True, font_weight='bold')
    plt.show()

if __name__ == "__main__":
    n_nodes = 100 # ノード数
    k_neighbors = 20 # 隣接ノード数
    p_rewire = 1 # 再配線確率

    # n_nodes = 20 # ノード数Watts
    # k_neighbors = 4 # 隣接ノード数(最初の接続数,rewireによって接続数は変化)
    # p_rewire = 0.3 # 再配線確率

    G = Watts_Strogats_small_world_graph(n_nodes, k_neighbors, p_rewire)
    graph_plot(G)

