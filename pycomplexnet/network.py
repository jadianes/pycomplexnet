
import numpy as np
from scipy.sparse import coo_matrix

class Network:
    """A directed graph representation of a Network
        The convention for the adjacency matrix is that, if adjacency_matrix[i,j]==1, then
        there is a link from node i to node j.
    """
 
    def __init__(self, adjacency_matrix, node_labels=None):
        self.adjacency_matrix = adjacency_matrix
        if node_labels is not None:
            self.node_labels = node_labels

    def from_pandas(links, from_column='from', to_column='to'):
        from_nodes = links[from_column]
        to_nodes = links[to_column]
        # get number of unique nodes and labels
        n_nodes = len(list(set(np.concatenate([from_nodes.values, to_nodes.values]))))
        all_nodes = []
        node_labels_t = {}
        node_i = 0
        # create adjacency matrix
        a = np.zeros((n_nodes,n_nodes))
        for from_node, to_node in zip(from_nodes, to_nodes):
            # this way of collecting unique nodes is a way to enforce 
            # the order given in the data frame, versus using list(set())
            # which doesn't guarantee the order
            if from_node not in all_nodes:
                all_nodes.append(from_node)
                node_labels_t[from_node] = node_i
                node_i += 1
            if to_node not in all_nodes:
                all_nodes.append(to_node)
                node_labels_t[to_node] = node_i
                node_i += 1
            # add link to adjacency matrix
            a[node_labels_t[from_node], node_labels_t[to_node]] = 1.0
            
        # create network
        return Network(
            coo_matrix(a), 
            {
                index:node for node, index in zip(all_nodes, range(0,n_nodes))
            })

    def toarray(self):
        return self.adjacency_matrix.toarray()
        
    def get_in_degree(self, node_i):
        """The in degree is the sum of all elements of the i_th column of the adjacency matrix"""
        return self.adjacency_matrix.getcol(node_i).sum()

    def get_in_degrees(self):
        """The list of in degrees is the sum of all elements of each row of the adjacency matrix"""
        return self.adjacency_matrix.sum(0)

    def get_out_degree(self, node_i):
        """The in degree is the sum of all elements of the i_th row of the adjacency matrix"""
        return self.adjacency_matrix.getrow(node_i).sum()

    def get_out_degrees(self):
        """The list of in degrees is the sum of all elements of each column of the adjacency matrix"""
        return self.adjacency_matrix.sum(1).flatten()

    def get_out_neighbourhood(self, node_i):
        """Get the list of nodes that node_i connects to, which are given by the i_th row of the adjacency matrix"""
        return self.adjacency_matrix.toarray()[node_i,:]

    def get_in_neighbourhood(self, node_i):
        """Get the list of nodes that connect with node_i, which are given by the i_th column of the adjacency matrix"""
        return self.adjacency_matrix.toarray()[:,node_i]

    def get_num_triangles(self, node_i):
        """The number of triangles starting and ending in node_i can be found in the diagonal of A^3"""
        A = self.toarray()
        A_3 = (A@A@A)
        if isinstance(node_i,list):
            return [int(A_3[i,i]) for i in node_i]
        else:
            return int(A_3[node_i,node_i])

    def get_joint_degree_dist(self, k_in, k_out):
        return (
            (1.0/self.adjacency_matrix.get_shape()[0])
            *(
                float(np.sum(
                    np.multiply(
                        self.get_in_degrees()==k_in,
                        self.get_out_degrees()==k_out
                    )
                ))
            )
        )

    def get_in_degree_dist(self, k_in):
        return (
            (1.0/self.adjacency_matrix.get_shape()[0])
            *(
                float(np.sum(
                    self.get_in_degrees()==k_in
                ))
            )
        )
    
    def get_out_degree_dist(self, k_out):
        return (
            (1.0/self.adjacency_matrix.get_shape()[0])
            *(
                float(np.sum(
                    self.get_out_degrees()==k_out
                ))
            )
        )

    def get_distance(self, node_i, node_j):
        steps = 1
        a = b = self.adjacency_matrix.toarray()
        while steps<a.shape[0]:
            if b[node_j,node_i] > 0:
                return steps
            b = np.matmul(b,a)
            steps+=1

        return np.Inf
