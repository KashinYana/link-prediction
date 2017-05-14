from graph_tool.all import *
import tools
import numpy as np
from sklearn.metrics import roc_auc_score

def calculate_auc_NMF(n_components, train_set, nodes, poss_set, neg_set, directed):
    from sklearn.decomposition import NMF
    model = NMF(n_components=n_components, init='random', random_state=0)
    matrix = tools.make_sparse_matrix(train_set, nodes, poss_set, directed=directed)
    features = tools.MFFeatures(model, matrix)
    X, Y = tools.make_dataset(poss_set, neg_set, [features.score])
    return roc_auc_score(Y, X)


def calculate_auc_PA(g, poss_set, neg_set):
    features = tools.TopologicalFeatures(g)
    X, Y = tools.make_dataset(poss_set, neg_set, [features.preferential_attachment])
    return roc_auc_score(Y, X)


def calculate_auc_CN(g, poss_set, neg_set):
    features = tools.TopologicalFeatures(g)
    X, Y = tools.make_dataset(poss_set, neg_set, [features.common_neighbors])
    return roc_auc_score(Y, X)


def calculate_auc_Adamic_Adar(g, poss_set, neg_set):
    features = tools.TopologicalFeatures(g)
    X, Y = tools.make_dataset(poss_set, neg_set, [features.Adamic_Adar_coefficient])
    return roc_auc_score(Y, X)


def calculate_auc_SVDS(n_components, train_set, nodes, poss_set, neg_set, directed):
    from scipy.sparse import linalg
    import numpy

    matrix = tools.make_sparse_matrix(train_set, nodes, poss_set, directed=directed)
    U, s, Vh = linalg.svds(matrix.asfptype(), k=n_components)
    U = U * s

    def score(u, w):
        return numpy.dot(U[u], Vh.T[w])

    X, Y = tools.make_dataset(poss_set, neg_set, [score])
    return roc_auc_score(Y, X)


def calculate_auc_directed(g, verbose, gap, poss_set, neg_set):
    groups = g.new_vertex_property("int")
    for u in g.vertices():
        groups[u] = int(u) % 2

    pos_directed = sfdp_layout(g, groups=groups, verbose=verbose, bipartite=True, gap=gap)
    
    features = tools.TopologicalFeatures(g, pos_directed, directed=True, gap=gap)
    X, Y = tools.make_dataset(poss_set, neg_set, [features.dist])
    return roc_auc_score(Y, X)
    
    
def calculate_auc_default(g, max_iter, poss_set, neg_set, p):
    pos_default = sfdp_layout(g, max_iter=max_iter, p=p)
    
    features = tools.TopologicalFeatures(g, pos_default)
    X, Y = tools.make_dataset(poss_set, neg_set, [features.dist])
    return roc_auc_score(Y, X)
    
    
def calculate_auc(train_set, nodes, poss_set, neg_set, auc, gap, verbose, directed, bipartite, max_iter, p):
    g = Graph(directed=False)
    
    if not directed:
        g.add_vertex(max(nodes) + 1)
        for edge in train_set:
            if edge not in poss_set:
                u, w = map(int, edge.split())
                g.add_edge(g.vertex(u), g.vertex(w))
    else:
        g.add_vertex(2*max(nodes) + 1)
        for edge in train_set:
            if edge not in poss_set:
                u, w = map(int, edge.split())
                g.add_edge(g.vertex(2*u - 1), g.vertex(2*w))
    
    auc["sfdp-default"].append(calculate_auc_default(g, max_iter, poss_set, neg_set, p))
    
    is_bi, part = graph_tool.topology.is_bipartite(g, partition=True)
    if bipartite and is_bi:
        groups = g.new_vertex_property("int")
        for u in g.vertices():
            groups[u] = int(part[u])
        left = "repulse-aliens"
        right = "repulse-aliens"
        pos_bip = sfdp_layout(g, groups=groups, verbose=verbose, 
                                  bipartite=True, bipartite_method=[left, right], gap=gap)
        features = tools.TopologicalFeatures(g, pos_bip, gap=gap)
        X, Y = tools.make_dataset(poss_set, neg_set, [features.dist])
        auc_name = "sfdp-bipartite"
        if auc_name not in auc:
            auc[auc_name] = []
            auc[auc_name].append(roc_auc_score(Y, X))
    if directed:
        if "sfdp-directed" not in auc:
            auc["sfdp-directed"] = []
        auc["sfdp-directed"].append(calculate_auc_directed(g, verbose, gap, poss_set, neg_set))
    
    auc["PA"].append(calculate_auc_PA(g, poss_set, neg_set))
    auc["CN"].append(calculate_auc_CN(g, poss_set, neg_set))
    auc["Adamic-Adar"].append(calculate_auc_Adamic_Adar(g, poss_set, neg_set))
    auc["NMF-10"].append(
        calculate_auc_NMF(10, train_set, nodes, poss_set, neg_set, directed))
    auc["svds-30"].append(
        calculate_auc_SVDS(30, train_set, nodes, poss_set, neg_set, directed))
    auc["NMF-30"].append(
        calculate_auc_NMF(30, train_set, nodes, poss_set, neg_set, directed))
    auc["svds-10"].append(
        calculate_auc_SVDS(10, train_set, nodes, poss_set, neg_set, directed))
    return auc


def cross_validation(file, N, k, gap=0, verbose=False, directed=False, bipartite=False, max_iter=0, comment='', p=2):
    auc = {
        "sfdp-default" : [],
        "PA" : [],
        "CN" : [],
        "Adamic-Adar" : [],
        "NMF-10" : [],
        "svds-10" : [],
        "NMF-30" : [],
        "svds-30" : [],
    }
    
    train_set, nodes, poss_set, neg_set = None, None, None, None
    poss_set_2, neg_set_2 =  None, None
    
    for i in range(k):
        train_set, nodes, poss_set, neg_set = None, None, None, None
        
        if bipartite:
            train_set, nodes, poss_set, neg_set = tools.sample_bipartite(file, N)
        else:
            train_set, nodes, poss_set, neg_set = tools.sample_structural(file, N, directed=directed)
        calculate_auc(train_set, nodes, poss_set, neg_set, auc, gap, verbose, directed, bipartite, max_iter, p)
    
    FIN = open('cross_validation', 'a')
    FIN.write('file:' + file + ' ' +
              'N: ' + str(N) + ' ' +
              'k: ' + str(k) + ' ' +
              'gap:' + str(gap) + ' ' +
              'verbose:' + str(verbose) + ' ' +
              'directed:' + str(directed) + ' ' +
              'bipartite:' + str(bipartite) + ' ' +
              'max_iter:' + str(max_iter) +
              'p:' + str(p) + '\n')
    if comment:
        FIN.write(comment + '\n')
    
    for x in auc:
        if not full:
            s = x + ' ' +  ": %0.2f (+/- %0.2f)" % (np.array(auc[x]).mean(), np.array(auc[x]).std() * 2)
        else:
            s = x + ' ' +  ": %0.8f (+/- %0.8f)" % (np.array(auc[x]).mean(), np.array(auc[x]).std() * 2)
        FIN.write(s + '\n')
        print s
    