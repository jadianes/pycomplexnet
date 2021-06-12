
import numpy as np

class Network:
    """A directed graph representation of a Network"""
 
    def __init__(self, adjacency_matrix):
        self.adjacency_matrix = adjacency_matrix

    def toarray(self):
        return self.adjacency_matrix.toarray()

    def get_in_degree(self, node_i):
        return self.adjacency_matrix.getcol(node_i).sum()

    def get_in_degrees(self):
        return self.adjacency_matrix.sum(0)

    def get_out_degree(self, node_i):
        return self.adjacency_matrix.getrow(node_i).sum()

    def get_out_degrees(self):
        return self.adjacency_matrix.sum(1).flatten()

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