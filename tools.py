def in_set(u, w, Set):
    return (u + ' ' + w in Set) or (w + ' ' + u in Set)

def read_train(file):
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
    return train_set, left_nodes, right_nodes

def sample_structural_poss(train_set, SIZE_POS, directed):
    import random
    poss_set = set(random.sample(train_set, SIZE_POS))
    return poss_set
    
def sample_structural_neg(train_set, nodes, SIZE_NEG, directed):
    import random
    neg_set = set()
    
    while len(neg_set) < SIZE_NEG:
        u_sample = random.sample(nodes, 100)
        w_sample = random.sample(nodes, 100)
        for (u, w) in zip(u_sample, w_sample):
            if u == w:
                continue
            u = str(u)
            w = str(w)
            if u + ' ' + w in train_set:
                continue
            if u + ' ' + w in neg_set:
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

def sample_structural(file, N, directed=False):
    train_set, left_nodes, right_nodes = read_train(file)
    nodes = left_nodes | right_nodes
    SIZE_POS = int(N * len(train_set) / 100.)
    SIZE_NEG = SIZE_POS
    poss_set = sample_structural_poss(train_set, SIZE_POS, directed)
    neg_set = sample_structural_neg(train_set, nodes, SIZE_NEG, directed)
    return train_set, nodes, poss_set, neg_set

def sample_bipartite(file, N):
    train_set, left_nodes, right_nodes = read_train(file)
    SIZE_POS = int(N * len(train_set) / 100.)
    SIZE_NEG = SIZE_POS
    poss_set = sample_structural_poss(train_set, SIZE_POS, directed=False)
    neg_set = sample_bipartite_neg(train_set, left_nodes, right_nodes, SIZE_NEG)
    return train_set, left_nodes | right_nodes, poss_set, neg_set

class TopologicalFeatures:
    def __init__(self, graph, pos=None, bipartite=False, gap=1):
        self.g = graph
        self.pos = pos
        self.gap = gap
        self.bipartite = bipartite
        
    def convert(self, u, w):
        return 2*u-1, 2*w
        
    def dist(self, u, w):
        if self.bipartite:
            u, w = self.convert(u, w)
        u = self.pos[self.g.vertex(u)]
        w = self.pos[self.g.vertex(w)]
        return -((u[0] - w[0])**2 + (u[1] - w[1])**2 + self.gap*self.gap)**0.5

    def preferential_attachment(self, u, w):
        if self.bipartite:
            u, w = self.convert(u, w)
        return self.g.vertex(u).out_degree()*self.g.vertex(w).out_degree()

    def common_neighbors(self, u, w):
        if self.bipartite:
            u, w = self.convert(u, w)
        return len(set.intersection(
            set(self.g.vertex(u).out_neighbours()), 
            set(self.g.vertex(w).out_neighbours())))

    def union_neighbors(self, u, w):
        if self.bipartite:
            u, w = self.convert(u, w)
        return len(
            set(self.g.vertex(u).out_neighbours()) | set(self.g.vertex(w).out_neighbours()))

    def Jaccards_coefficient(self, u, w):
        if self.bipartite:
            u, w = self.convert(u, w)
        if union_neighbors(u, w) == 0:
            return 0
        return 1.0 * self.common_neighbors(u, w) / self.union_neighbors(u, w)

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

def make_sparse_matrix(train_set, nodes, poss_set=set()):
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
        row.append(w)
        col.append(u)
        data.append(1)
        data.append(1)
    return coo_matrix((data, (row, col)), shape=(n, n))

class MFFeatures:
    def __init__(self, model, matrix):
        self.W = model.fit_transform(matrix)
        self.H = model.components_;
        
    def score(self, u, w):
        import numpy
        return numpy.dot(self.W[u], self.H.T[w])
