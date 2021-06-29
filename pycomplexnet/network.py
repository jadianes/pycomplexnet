
import numpy as np
from scipy.sparse import coo_matrix
import itertools

class Network:
    """A directed graph representation of a Network
        The convention for the adjacency matrix is that, if adjacency_matrix[i,j]==1, then
        there is a link from node i to node j.
    """
 
    def __init__(self, adjacency_matrix, node_labels=None):
        self.adjacency_matrix = adjacency_matrix
        if node_labels is None:
            node_labels=[node_i for node_i in range(0,adjacency_matrix.shape[0])]
        self.node_labels = {
            index:node for node, index in zip(node_labels, range(0,len(node_labels)))
        }
        self.node_labels_map = {
            node:index for node, index in zip(node_labels, range(0,len(node_labels)))
        }


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
        return Network(coo_matrix(a), all_nodes)

    def toarray(self):
        return self.adjacency_matrix.toarray()
        
    def get_in_degree(self, node_i):
        """The in degree is the sum of all elements of the i_th column of the adjacency matrix"""
        try:
            node_i = self.node_labels_map[node_i]
        except:
            pass
        return self.adjacency_matrix.getcol(node_i).sum()

    def get_in_degrees(self):
        """The list of in degrees is the sum of all elements of each row of the adjacency matrix"""
        return self.adjacency_matrix.sum(0)

    def get_out_degree(self, node_i):
        """The in degree is the sum of all elements of the i_th row of the adjacency matrix"""
        try:
            node_i = self.node_labels_map[node_i]
        except:
            pass
        return self.adjacency_matrix.getrow(node_i).sum()

    def get_out_degrees(self):
        """The list of in degrees is the sum of all elements of each column of the adjacency matrix"""
        return self.adjacency_matrix.sum(1).flatten()

    def get_out_neighbourhood(self, node_i):
        """Get the list of nodes that node_i connects to, which are given by the i_th row of the adjacency matrix"""
        try:
            node_i = self.node_labels_map[node_i]
        except:
            pass
        return self.adjacency_matrix.toarray()[node_i,:]

    def get_in_neighbourhood(self, node_i):
        """Get the list of nodes that connect with node_i, which are given by the i_th column of the adjacency matrix"""
        try:
            node_i = self.node_labels_map[node_i]
        except:
            pass
        return self.adjacency_matrix.toarray()[:,node_i]

    def get_num_triangles(self, node_i):
        """The number of triangles starting and ending in node_i can be found in the diagonal of A^3"""
        A = self.toarray()
        A_3 = (A@A@A)
        if isinstance(node_i,list):
            try:
                node_i = [self.node_labels_map[i] for i in node_i] 
            except:
                pass
            return [int(A_3[i,i]) for i in node_i]
        else:
            try:
                node_i = self.node_labels_map[node_i]
            except:
                pass
            return int(A_3[node_i,node_i])

    def get_reciprocal_degree(self, node_i):
        A = self.toarray()
        if isinstance(node_i,list):
            try:
                node_i = [self.node_labels_map[i] for i in node_i] 
            except:
                pass
            return [sum(A[i,:]*A[:,i]) for i in node_i]
        else:
            try:
                node_i = self.node_labels_map[node_i]
            except:
                pass
            return sum(A[node_i,:]*A[:,node_i])

    def get_clustering_coeff(self, node_i):
        t = self.get_num_triangles(node_i)
        deg_tot = self.get_in_degree(node_i) + self.get_out_degree(node_i)
        deg_recip = self.get_reciprocal_degree(node_i)
        return float(t) / (deg_tot*(deg_tot-1.0) - 2.0 * deg_recip)

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
        return self.get_shortest_paths(node_i, node_j)[0]

    def get_average_distance(self, node_i):
        N = self.adjacency_matrix.shape[0]
        return sum([
            self.get_distance(node_i, self.node_labels[j]) 
            for j in range(0,N)
        ]) / N

    def get_closeness_centrality(self, node_i):
        return 1.0/self.get_average_distance(node_i)

    def get_shortest_paths(self, node_i, node_j):
        """The number of shortest paths between two nodes, if any, is
        given by multiplying the adjacency matrix by itself.
        Returns: (length, number of paths)
        """
        try:
            node_i = self.node_labels_map[node_i]
        except:
            pass
        try:
            node_j = self.node_labels_map[node_j]
        except:
            pass
        # paths to self
        if node_i==node_j:
            return (0,1)
        # path to different nodes
        steps = 1
        a = b = self.adjacency_matrix.toarray()
        while steps<a.shape[0]:
            if b[node_i,node_j] > 0:
                return (steps, int(b[node_i,node_j]))
            b = b@a
            steps+=1

        return (np.Inf, 0)

    def get_shortest_paths_by(self, node_j, node_l, node_i):
        """We interested in the number of shortest paths between node_j and
        node_l passing through node_i"""
        if (node_j==node_l) and (node_i!=node_j):
            return (np.Inf, 0)
        elif (node_i==node_j) or (node_i==node_l):
            return self.get_shortest_paths(node_j, node_l)
        else:
            # d_j_l = length of shortest path from node_j to node_l
            d_j_l = self.get_shortest_paths(node_j, node_l)[0]
            # d_j_i = length of shortest path from node_j to node_i
            d_j_i = self.get_shortest_paths(node_j, node_i)[0]
            # check we can still get to l faster
            if d_j_i < d_j_l:
                # d_i_l = length of shortest path from node_i to node_l
                d_i_l = self.get_shortest_paths(node_i, node_l)[0]
                # check i actually is part of the shortest paths (it can't be shorter)
                if (d_j_i + d_i_l) == d_j_l:
                    # the result is the number of combinations
                    return (d_j_l, self.get_shortest_paths(node_j, node_i)[1] * self.get_shortest_paths(node_i, node_l)[1])
                else:
                    return (np.Inf, 0)
            else:
                return (np.Inf, 0)

    def get_betweeness_centrality(self, node_i):
        """y_i(A) = 
            sum_j  
                sum_l 
                    num_shortest_paths_by(j,l,i) / num_shortest_paths(j,l) 
        """
        N = self.adjacency_matrix.shape[0]
        num_shortest_paths = [
            self.get_shortest_paths(self.node_labels[j],self.node_labels[l])[1] 
            for j,l in itertools.product(range(0,N),range(0,N))
        ]
        num_shortest_paths_by = [
            self.get_shortest_paths_by(self.node_labels[j],self.node_labels[l], node_i)[1] 
            for j,l in itertools.product(range(0,N),range(0,N))
        ]
        return sum([
            n / g if (g!=0) else 0
            for n,g in zip(num_shortest_paths_by,num_shortest_paths)
        ])