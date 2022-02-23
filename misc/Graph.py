""" a collection of doubly stochastic matrix for connected graph """
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
""" Graph package """
__license__ = 'MIT'
__author__ = 'Shixiang Chen'
__email__ = 'chenshxiang@gmail.com'


class Graph:
    """
    Parameters: graph_type = 'ER' or 'Ring', default = 'Ring'
                weighted_type = 'max_degree' or 'metropolis_hastings'
                N: number of nodes
                plot_g: whether plot the graph
                p: connection probability for ER graph
    Return: build_graph(): return peer
            generate_weighted_mat(): return the doubly stochastic weighted matrix W
    """
    def __init__(self, graph_type='Ring', weighted_type='max_degree', N=2, plot_g=False, p=0):
        self.graph_type = graph_type
        self.weighted_type = weighted_type
        self.N = N
        self.plot_g = plot_g
        self.G = None
        self.L = None   # Laplacian  matrix
        self.degree = 0    # degree of each node
        self.peer = None    # neighborhood of each node
        self.p = p  # connection probability for ER graph
        if graph_type:
            if self.N == 1:
                self.peer = np.array([0])
            else:
                self.peer = self.build_graph()
        if weighted_type:
            if self.N == 1:
                self.W = np.array([[1]])
            else:
                self.W = self.generate_weighted_mat()

    def get_degree_peer_adj(self):
        self.peer = [[] for _ in range(self.N)]
        for idx, val in np.ndenumerate(self.L):
            if val == -1:
                self.peer[idx[0]].append(idx[1])
        self.degree = [len(a) for a in self.peer]
        # print(self.degree)

    def ring_graph(self):
        self.G = nx.cycle_graph(self.N)
        self.L = nx.laplacian_matrix(self.G).toarray()
        self.get_degree_peer_adj()
        print(f'The ring graph is generated, size of {self.N}')
        return self.peer

    def star_graph(self):
        self.G = nx.star_graph(self.N-1)
        self.L = nx.laplacian_matrix(self.G).toarray()
        self.get_degree_peer_adj()
        print(f'The star graph is generated, size of {self.N}')
        return self.peer

    def er_graph(self):
        """ Erdos Renyi graph with N nodes, connective probability p"""
        while True:
            self.G = nx.erdos_renyi_graph(self.N, self.p)
            self.L = nx.laplacian_matrix(self.G).toarray()
            _, s, _ = np.linalg.svd(self.L)
            s.sort()
            if s[1] > 0.01:
                print(f'The Erdos-Renyi graph is generated. Algebraic Connectivity: {s[1]:.3f}, size of {self.N}')
                break
        self.get_degree_peer_adj()
        return self.peer

    def complete(self):
        """ complete graph with N nodes """
        self.G = nx.complete_graph(self.N)
        self.L = nx.laplacian_matrix(self.G).toarray()
        self.get_degree_peer_adj()
        print(f'The complete graph is generated, size of {self.N}')
        return self.peer

    def laplacian_based(self):
        d_max = max(self.degree)
        W = np.eye(self.N) - self.L / (d_max+1)
        print(f'Maximum-degree matrix is generated')
        return W

    def metropolis_hastings(self):
        W = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                if i != j and j in self.peer[i]:
                    W[i][j] = 1 / max(self.degree[i] + 1, self.degree[j] + 1)
            W[i][i] = 1 - sum(W[i])
        print(f'Metropolis Hastings matrix is generated')
        return W

    def lazy_metropolis(self):
        W = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                if i != j and j in self.peer[i]:
                    W[i][j] = 0.5 / max(self.degree[i], self.degree[j])
            W[i][i] = 1 - sum(W[i])
        print(f'Metropolis Hastings matrix is generated')
        return W

    def build_graph(self):
        switcher = {
            'ER': self.er_graph,
            'Ring': self.ring_graph,
            'star': self.star_graph,
            'complete': self.complete
        }
        method = switcher.get(self.graph_type, "Invalid graph")()
        # method()
        if self.plot_g:
            nx.draw(self.G)
            txt = "The" + ' ' + self.graph_type + ' ' + 'graph' + ',' + ' ' + 'n = ' + str(self.N)
            txt += ', p=' + str(self.p) if self.graph_type == 'ER' else None
            plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=10)
            import os
            if not os.path.isdir("Graph"):
                os.makedirs("Graph")
            save_path = '../Graph'
            plt.savefig(os.path.join(save_path, self.graph_type + str(self.N) + str(self.p) + '.pdf'), format='pdf', dpi=3000)
            plt.show()

        return self.peer

    def generate_weighted_mat(self):
        switcher = {
            'Laplacian-based': self.laplacian_based,
            'metropolis_hastings': self.metropolis_hastings,
            'lazy_metropolis': self.lazy_metropolis
        }
        method = switcher.get(self.weighted_type, "Invalid type of weighted matrix")
        return method()


if __name__ == '__main__':
    graph_type = ['Ring', 'ER', 'star', 'complete']
    weighted_type = ['Laplacian-based', 'lazy_metropolis', 'metropolis_hastings']
    g = Graph(graph_type=graph_type[1], weighted_type=weighted_type[2],
              N=10, p=0.5, plot_g=True)
    peer = g.peer
    W = g.W
    print(W)
    # _, s, _ = np.linalg.svd(W)
    # print(s)






