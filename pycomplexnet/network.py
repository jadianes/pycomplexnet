
class Network:
    """A Complex Network representation"""
 
    def __init__(self, adjacency_matrix):
        self.adjacency_matrix = adjacency_matrix

    def get_in_degree(self, node_i):
        return self.adjacency_matrix.getcol(node_i).sum()

    def get_in_degrees(self):
        return self.adjacency_matrix.sum(0)

    def get_out_degree(self, node_i):
        return self.adjacency_matrix.getrow(node_i).sum()

    def get_out_degrees(self):
        return self.adjacency_matrix.sum(1).flatten()

    def toarray(self):
        return self.adjacency_matrix.toarray()