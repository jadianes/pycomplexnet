from .network import Network
from scipy.sparse import coo_matrix, rand
import numpy as np

def disconnected_network(n_nodes):
    adjacency_matrix = coo_matrix((n_nodes,n_nodes))
    return Network(adjacency_matrix)

def random_network(n_nodes, density):
    adjacency_matrix = rand(n_nodes, n_nodes, density=density, format='coo').rint()
    return Network(adjacency_matrix)

def star_network(n_nodes, center_node):
    a = np.zeros((n_nodes,n_nodes))
    a[center_node,:] = 1.0
    a[:,center_node] = 1.0
    a[center_node,center_node] = 0.0
    adjacency_matrix = coo_matrix(a)
    return Network(adjacency_matrix)