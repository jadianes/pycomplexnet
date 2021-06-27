from os import sep
from .network import Network
from scipy.sparse import coo_matrix, rand
import numpy as np
import pandas as pd
import pkgutil
import io

def disconnected_network(n_nodes):
    adjacency_matrix = coo_matrix((n_nodes,n_nodes))
    return Network(adjacency_matrix)

def random_network(n_nodes, p):
    adjacency_matrix = ( rand(n_nodes, n_nodes, density=1, format='coo') > (1.0-p) ).astype('float')
    return Network(adjacency_matrix)

def star_network(n_nodes, center_node):
    a = np.zeros((n_nodes,n_nodes))
    a[center_node,:] = 1.0
    a[:,center_node] = 1.0
    a[center_node,center_node] = 0.0
    adjacency_matrix = coo_matrix(a)
    return Network(adjacency_matrix)

def zachary_karate_club_77():
    # read data from resources to create data frame of links
    edges_file = pkgutil.get_data(__name__, "data/zkcc-77/karate_edges_77.txt").decode()
    edges_df = pd.read_csv(io.StringIO(edges_file), sep='\s', header=None, names=['from','to'])
    
    return Network.from_pandas(edges_df)