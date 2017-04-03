from graph_tool.all import *
import tools
import numpy as np

def calculate_auc(train_set, nodes, poss_set, neg_set, auc):
    g = Graph(directed=False)
    g.add_vertex(max(nodes) + 1)
    
    for edge in train_set:
        if edge not in poss_set:
            u, w = map(int, edge.split())
            g.add_edge(g.vertex(u), g.vertex(w))
    
    is_bi, part = graph_tool.topology.is_bipartite(g, partition=True)
    if (not is_bi):
        print "Error"
        
    groups = g.new_vertex_property("int")
    
    for u in g.vertices():
        groups[u] = part[u]
    
    pos_default = sfdp_layout(g)
    pos_bip = sfdp_layout(g, groups=groups, bipartite=True)
    
    from sklearn.metrics import roc_auc_score
    features = tools.TopologicalFeatures(g, pos_default, gap=0)
    X, Y = tools.make_dataset(poss_set, neg_set, 
                        [features.dist])
    auc["sfdp-default"].append(roc_auc_score(Y, X))
    
    features = tools.TopologicalFeatures(g, pos_bip, gap=0)
    X, Y = tools.make_dataset(poss_set, neg_set, 
                    [features.dist])
    auc["sfdp-bipartite-simple"].append(roc_auc_score(Y, X))
    
    features = tools.TopologicalFeatures(g, pos_bip)
    X, Y = tools.make_dataset(poss_set, neg_set, 
                    [features.preferential_attachment])
    auc["PA"].append(roc_auc_score(Y, X))
    
    from sklearn.decomposition import NMF
    model = NMF(n_components=10, init='random', random_state=0)
    matrix = tools.make_sparse_matrix(train_set, nodes, poss_set)
    features = tools.MFFeatures(model, matrix)
    X, Y = tools.make_dataset(poss_set, neg_set, 
                        [features.score])
    auc["NMF-10"].append(roc_auc_score(Y, X))
    
    from scipy.sparse import linalg
    import numpy

    matrix = tools.make_sparse_matrix(train_set, nodes, poss_set)
    U, s, Vh = linalg.svds(matrix.asfptype(), k=30)

    def score(u, w):
        return numpy.dot(U[u] * s, Vh.T[w])

    features = tools.MFFeatures(model, matrix)
    X, Y = tools.make_dataset(poss_set, neg_set, 
                        [score])
    auc["svds-30"].append(roc_auc_score(Y, X))
    
    return auc


def cross_validation(file, N, k):
    auc = {
        "sfdp-default" : [],
        "sfdp-bipartite-simple" : [],
        "PA" : [],
        "NMF-10" : [],
        "svds-30" : [],
    }
    for i in range(k):
        train_set, nodes, poss_set, neg_set = tools.sample_bipartite(file, N)
        calculate_auc(train_set, nodes, poss_set, neg_set, auc)
    
    FIN = open('cross_validation', 'a')
    FIN.write(file + ' ' + str(N) + ' ' + str(k) + '\n')
    for x in auc:
        s = x + ' ' +  ": %0.2f (+/- %0.2f)" % (np.array(auc[x]).mean(), np.array(auc[x]).std() * 2)
        FIN.write(s + '\n')
        print s
    