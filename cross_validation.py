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


def calculate_auc_SVDS(n_components, train_set, nodes, poss_set, neg_set, directed):
    from scipy.sparse import linalg
    import numpy

    matrix = tools.make_sparse_matrix(train_set, nodes, poss_set, directed=directed)
    U, s, Vh = linalg.svds(matrix.asfptype(), k=n_components)

    def score(u, w):
        return numpy.dot(U[u] * s, Vh.T[w])

    X, Y = tools.make_dataset(poss_set, neg_set, [score])
    return roc_auc_score(Y, X)


def calculate_auc_directed(g, verbose, gap, poss_set, neg_set):
    groups = g.new_vertex_property("int")
    for u in g.vertices():
        groups[u] = int(u) % 2

    pos_directed = sfdp_layout(g, groups=groups, verbose=verbose, bipartite=True, gap=gap)

    print "gap", gap
    
    features = tools.TopologicalFeatures(g, pos_directed, directed=True, gap=gap)
    X, Y = tools.make_dataset(poss_set, neg_set, [features.dist])
    return roc_auc_score(Y, X)
    
    
def calculate_auc_default(g, max_iter, poss_set, neg_set):
    pos_default = sfdp_layout(g, max_iter=max_iter)
    
    features = tools.TopologicalFeatures(g, pos_default)
    X, Y = tools.make_dataset(poss_set, neg_set, [features.dist])
    return roc_auc_score(Y, X)
    
    
def calculate_auc(train_set, nodes, poss_set, neg_set, auc, gap, verbose, directed, bipartite, max_iter):
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
    
    auc["sfdp-default"].append(calculate_auc_default(g, max_iter, poss_set, neg_set))
    
    is_bi, part = graph_tool.topology.is_bipartite(g, partition=True)
    if bipartite and is_bi:
        groups = g.new_vertex_property("int")
        for u in g.vertices():
            groups[u] = int(part[u])
        for left in ["repulse-fellows", "repulse-aliens"]:
            for right in ["repulse-fellows", "repulse-aliens"]:
                pos_bip = sfdp_layout(g, groups=groups, verbose=verbose, 
                                      bipartite=True, bipartite_method=[left, right], gap=gap)
                features = tools.TopologicalFeatures(g, pos_bip, gap=gap)
                X, Y = tools.make_dataset(poss_set, neg_set, [features.dist])
                auc["sfdp-bipartite-" + left+right].append(roc_auc_score(Y, X))
    if directed:
        auc["sfdp-directed"].append(calculate_auc_directed(g, verbose, gap, poss_set, neg_set))
    
    auc["PA"].append(calculate_auc_PA(g, poss_set, neg_set))
    auc["NMF-10"].append(
        calculate_auc_NMF(10, train_set, nodes, poss_set, neg_set, directed))
    auc["svds-30"].append(
        calculate_auc_SVDS(30, train_set, nodes, poss_set, neg_set, directed))
    
    return auc


def cross_validation(file, N, k, gap=0, verbose=False, directed=False, bipartite=False, max_iter=0):
    auc = {
        "sfdp-default" : [],
        "PA" : [],
        "NMF-10" : [],
        "svds-30" : [],
        "sfdp-directed" :[],
    }
    for left in ["repulse-fellows", "repulse-aliens", "repulse-all"]:
            for right in ["repulse-fellows", "repulse-aliens", "repulse-all"]:
                auc["sfdp-bipartite-" + left+right] = []
    
    for i in range(k):
        train_set, nodes, poss_set, neg_set = None, None, None, None
        
        if bipartite:
            train_set, nodes, poss_set, neg_set = tools.sample_bipartite(file, N)
        else:
            train_set, nodes, poss_set, neg_set = tools.sample_structural(file, N, directed=directed)
        calculate_auc(train_set, nodes, poss_set, neg_set, auc, gap, verbose, directed, bipartite, max_iter)
    
    FIN = open('cross_validation', 'a')
    FIN.write(file + ' ' + str(N) + ' ' + str(k) + '\n')
    for x in auc:
        s = x + ' ' +  ": %0.2f (+/- %0.2f)" % (np.array(auc[x]).mean(), np.array(auc[x]).std() * 2)
        FIN.write(s + '\n')
        print s
    