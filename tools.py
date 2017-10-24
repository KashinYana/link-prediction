from graph_tool.all import *
from save_sample import safe_sample_edges
from graph_tool import topology


def in_set(u, w, Set):
    return (u + ' ' + w in Set) or (w + ' ' + u in Set)


def choose_giant_component(edges, nodes):
    g = Graph(directed=False)
    g.add_vertex(max(nodes) + 1)
    for edge in edges:
        u, w = map(int, edge.split())
        g.add_edge(g.vertex(u), g.vertex(w))
    labels = topology.label_largest_component(g)
    u = GraphView(g, vfilt=labels)
    new_nodes = set()
    new_edges = set()
    for e in u.edges():
        new_nodes.add(int(e.source()))
        new_nodes.add(int(e.target()))
        new_edges.add(str(e.source()) + ' ' + str(e.target()))
    return new_edges, new_nodes


def rename(edges, nodes):
    d = {}
    i = 0
    for node in nodes:
        d[node] = i
        i += 1
    new_edges = set()
    new_nodes = set(range(i))
    for edge in edges:
        u, w = map(int, edge.strip().split())
        new_edges.add(str(d[u]) + ' ' + str(d[w]))
    return new_edges, new_nodes


def read_train(file, directed=False):
    train_set = set()
    left_nodes = set()
    right_nodes = set()
    
    FIN = open(file, 'r')
    for line in FIN:
        train_set.add(line.strip())
        u, w = line.strip().split()
        left_nodes.add(int(u))
        right_nodes.add(int(w))
    FIN.close()
    
    if directed:
        return train_set, left_nodes, right_nodes
        
    train_set_undirected = set()
    f = False
    for e in train_set:
        u, w = map(int, e.split())
        if ((str(u) + ' ' + str(w) not in train_set_undirected)
            and (str(w) + ' ' + str(u) not in train_set_undirected)):
            train_set_undirected.add(str(u) + ' ' + str(w))
        if (str(w) + ' ' + str(u)) in train_set:
            f = True
    
    if f:
        print('It was bug')
    return train_set_undirected, left_nodes, right_nodes

def sample_structural_poss(train_set, SIZE_POS, directed=False, sparse=False, nodes=None):
    import random
    
    if not sparse:
        return  set(random.sample(train_set, SIZE_POS))
    else:
        train_set = [tuple(map(int, edge.split())) for edge in train_set]
        nodes_r, edges_r, sampled_edges_r = safe_sample_edges(nodes, train_set, SIZE_POS)
        return [str(e[0]) + ' ' + str(e[1]) for e in sampled_edges_r]
        
        g = Graph(directed=False)
        g.add_vertex(max(nodes) + 1)
        for edge in train_set:
            u, w = map(int, edge.split())
            g.add_edge(g.vertex(u), g.vertex(w))
            
        poss_set = set()
        while len(poss_set) < SIZE_POS:
            updates = 0
            samples = set(random.sample(train_set, SIZE_POS))
            for sample in samples:
                u, w = map(int, sample.split())
                if (g.vertex(u).out_degree() > 1) and (g.vertex(w).out_degree() > 1) and g.edge(g.vertex(u), g.vertex(w)):
                    poss_set.add(str(u) + ' ' + str(w))
                    g.remove_edge(g.edge(g.vertex(u), g.vertex(w)))
                    updates += 1
            if updates == 0:
                break
       
        if len(poss_set) > SIZE_POS:
            poss_set = set(random.sample(poss_set, SIZE_POS))
        return poss_set      
    
def sample_structural_neg(train_set, nodes, SIZE_NEG, directed):
    import random
    
    DIFFICULT_EDGE_RATE = 0.5
    neg_set = set()
    difficult_edges = []
    
    if directed:
        for edge in train_set:
            u, w, = edge.split()
            if w + ' ' + u not in train_set:
                difficult_edges.append(w + ' ' + u)

        neg_set = set(random.sample(difficult_edges, 
            int(min(DIFFICULT_EDGE_RATE * SIZE_NEG, len(difficult_edges))) ))
    
    while len(neg_set) < SIZE_NEG:
        u_sample = random.sample(nodes, 1000)
        w_sample = random.sample(nodes, 1000)
        for (u, w) in zip(u_sample, w_sample):
            if u == w:
                continue
            u = str(u)
            w = str(w)
            if directed:
                if u + ' ' + w in train_set:
                    continue
                if u + ' ' + w in neg_set:
                    continue
            else:
                if in_set(u, w, train_set):
                    continue
                if in_set(u, w, neg_set):
                    continue
            neg_set.add(u + ' ' + w)
            
    if len(neg_set) > SIZE_NEG:
        neg_set = set(random.sample(neg_set, SIZE_NEG))
    return neg_set

def sample_bipartite_neg(train_set, left_nodes, right_nodes, SIZE_NEG):
    import random
    neg_set = set()
    while len(neg_set) < SIZE_NEG:
        u_sample = random.sample(left_nodes, 100)
        w_sample = random.sample(right_nodes, 100)
        for (u, w) in zip(u_sample, w_sample):
            if u == w:
                continue
            u = str(u)
            w = str(w)
            if in_set(u, w, train_set):
                continue
            if in_set(u, w, neg_set):
                continue
            neg_set.add(u + ' ' + w)
    if len(neg_set) > SIZE_NEG:
        neg_set = set(random.sample(neg_set, SIZE_NEG))
    return neg_set

def sample_structural(file, N, directed=False, sparse=False):
    train_set, left_nodes, right_nodes = read_train(file, directed=directed)
    nodes = left_nodes | right_nodes
    
    train_set, nodes = choose_giant_component(train_set, nodes)
    train_set, nodes = rename(train_set, nodes)
    
    SIZE_POS = int(N * len(train_set) / 100.)
    SIZE_NEG = SIZE_POS
    poss_set = sample_structural_poss(train_set, SIZE_POS, directed, sparse, nodes)
    neg_set = sample_structural_neg(train_set, nodes, SIZE_NEG, directed)
    return train_set, nodes, poss_set, neg_set

def sample_bipartite(file, N):
    train_set, left_nodes, right_nodes = read_train(file, directed=False)
    SIZE_POS = int(N * len(train_set) / 100.)
    SIZE_NEG = SIZE_POS
    poss_set = sample_structural_poss(train_set, SIZE_POS, directed=False, sparse=False)
    neg_set = sample_bipartite_neg(train_set, left_nodes, right_nodes, SIZE_NEG)
    return train_set, left_nodes | right_nodes, poss_set, neg_set

class TopologicalFeatures:
    def __init__(self, graph, pos=None, directed=False, gap=0):
        self.g = graph
        self.pos = pos
        self.gap = gap
        self.directed = directed
        
    def convert(self, u, w):
        return 2*u+1, 2*w
        
    def dist(self, u, w):
        #print('dist', u, w)
        if self.directed:
            u, w = self.convert(u, w)
        #print(u, w)
        u = self.pos[self.g.vertex(u)]
        w = self.pos[self.g.vertex(w)]
        
        #print(-((u[0] - w[0])**2 + (u[1] - w[1])**2 + self.gap*self.gap)**0.5)
        return -((u[0] - w[0])**2 + (u[1] - w[1])**2 + self.gap*self.gap)**0.5

    def preferential_attachment(self, u, w):
        if self.directed:
            u, w = self.convert(u, w)
        return self.g.vertex(u).out_degree()*self.g.vertex(w).out_degree()

    def common_neighbors(self, u, w):
        if self.directed:
            u, w = self.convert(u, w)
        return len(set.intersection(
            set(self.g.vertex(u).out_neighbours()), 
            set(self.g.vertex(w).out_neighbours())))

    def union_neighbors(self, u, w):
        if self.directed:
            u, w = self.convert(u, w)
        return len(
            set(self.g.vertex(u).out_neighbours()) | set(self.g.vertex(w).out_neighbours()))

    def Jaccards_coefficient(self, u, w):
        if self.directed:
            u, w = self.convert(u, w)
        if union_neighbors(u, w) == 0:
            return 0
        return 1.0 * self.common_neighbors(u, w) / self.union_neighbors(u, w)
    
    def Adamic_Adar_coefficient(self, u, w):
        import numpy as np
        if self.directed:
            u, w = self.convert(u, w)
        CN_set = set.intersection(
            set(self.g.vertex(u).out_neighbours()), 
            set(self.g.vertex(w).out_neighbours()))
        score = 0
        for z in CN_set:
            if self.g.vertex(z).out_degree() == 1:
                print u, w, z
            score += 1. / np.log(self.g.vertex(z).out_degree())
        return score

def make_dataset(poss_set, neg_set, functs):
    import numpy
    X = []
    Y = []
    for line in poss_set:
        u, w = map(int, line.split())
        x = []
        for func in functs:
            x.append(func(u, w))
        X.append(x)
        Y.append(1)
    
    for line in neg_set:
        u, w = map(int, line.split())
        x = []
        for func in functs:
            x.append(func(u, w))
        X.append(x)
        Y.append(0)
        
    X = numpy.array(X)
    Y = numpy.array(Y)
    return X, Y

def make_sparse_matrix(train_set, nodes, poss_set=set(), directed=False):
    n = max(nodes) + 1
    from scipy.sparse import coo_matrix
    row = []
    col = []
    data = []
    for line in train_set:
        if line in poss_set:
            continue
        u, w = map(int, line.split())
        row.append(u)
        col.append(w)
        data.append(1)
        if not directed:
            row.append(w)
            col.append(u)
            data.append(1)
    return coo_matrix((data, (row, col)), shape=(n, n))

class MFFeatures:
    def __init__(self, model, matrix):
        self.W = model.fit_transform(matrix)
        self.H = model.components_;
        
    def score(self, u, w):
        import numpy
        return numpy.dot(self.W[u], self.H.T[w])
    
    
class Node2VecFeatures:
    def __init__(self, d):
        self.d = d
        
    def score(self, u, w):
        import numpy
        return numpy.dot(self.d[u], self.d[w])
