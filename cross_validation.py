from graph_tool.all import *
import tools
import numpy as np
import json
from sklearn.metrics import roc_auc_score
import os


def calculate_auc_NMF(n_components, train_set, nodes, poss_set, neg_set, directed):
    from sklearn.decomposition import NMF
    model = NMF(n_components=n_components, init='random', random_state=0)
    matrix = tools.make_sparse_matrix(train_set, nodes, poss_set, directed=directed)
    features = tools.MFFeatures(model, matrix)
    X, Y = tools.make_dataset(poss_set, neg_set, [features.score])
    return roc_auc_score(Y, X)


def calculate_auc_PA(g, poss_set, neg_set):
    features = tools.TopologicalFeatures(g)
    edges = [e for e in g.edges()]
    print('real number edges', len(edges))
    poss_set = set(poss_set)
    neg_set = set(neg_set)
    for e in edges:
        e1 = str(e.source()) + ' ' + str(e.target())
        e2 = str(e.target()) + ' ' + str(e.source())
        if e1 in poss_set or e2 in poss_set:
            print('poss fails')
        if e1 in neg_set or e2 in neg_set:
            print('neg_set fails')
            
    X, Y = tools.make_dataset(poss_set, neg_set, [features.preferential_attachment])
    print('len X', len(X))
    print('len Y', len(Y))
    print('sum Y', sum(Y))
    print('describe', X[:len(X)/2].mean(), Y[:len(X)/2].mean())
    print('describe', X[len(X)/2:].mean(), Y[len(X)/2:].mean())
    print('describe', X.mean())
    
    return roc_auc_score(Y, X)


def calculate_auc_CN(g, poss_set, neg_set):
    features = tools.TopologicalFeatures(g)
    X, Y = tools.make_dataset(poss_set, neg_set, [features.common_neighbors])
    return roc_auc_score(Y, X)


def calculate_auc_Node2Vec(g, poss_set, neg_set, d=10):
    fout = open('node2vec.in', 'w')
    for e in g.edges():
        fout.write(str(e.source()) + ' ' + str(e.target()) + '\n')
    fout.close()
    
    print os.system('python ../node2vec/src/main.py --input node2vec.in --output node2vec.out ' +
          '--dimensions %d --walk-length 80 --p 1 --iter 1' % d)
    
    w2v = {}
    with open('node2vec.out') as fin:
        print(fin.readline())
        for line in fin:
            line = list(map(float, line.strip().split()))
            w2v[line[0]] = np.array(line[1:])
    print('len w2v', len(w2v))
    
    features = tools.Node2VecFeatures(w2v)
    X, Y = tools.make_dataset(poss_set, neg_set, [features.score])
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
    
    
def calculate_auc_default(g, max_iter, poss_set, neg_set, p, directed):
    pos_default = sfdp_layout(g, max_iter=max_iter, p=p)
    
    features = tools.TopologicalFeatures(g, pos_default, directed=directed)
    X, Y = tools.make_dataset(poss_set, neg_set, [features.dist])
    return roc_auc_score(Y, X)


def convert_set(edges, g):
    edges_copy = []
    
    for edge in edges:
        u, w = map(int, edge.split())
        if g.vertex(w).out_degree() > 0 and g.vertex(u).out_degree() / g.vertex(w).out_degree() < 0.5:
            edges_copy.append(edge)
        else:
            edges_copy.append(str(w) + ' ' + str(u))  
            
    return edges_copy


def add_auc(auc, name, value):
    if name not in auc:
        auc[name] = []
    auc[name].append(value)
    
    
def calculate_auc(train_set, nodes, poss_set, neg_set, auc, gap, verbose, directed, bipartite, max_iter, p):
    g = Graph(directed=False)
    
    #if not directed:
    g.add_vertex(max(nodes) + 1)
    for edge in train_set:
        if edge not in poss_set:
            u, w = map(int, edge.split())
            g.add_edge(g.vertex(u), g.vertex(w))
    
    
    #add_auc(auc, 'Node2Vec-5', calculate_auc_Node2Vec(g, poss_set, neg_set, 5))
    #add_auc(auc, 'Node2Vec-10', calculate_auc_Node2Vec(g, poss_set, neg_set, 10))
    #add_auc(auc, 'Node2Vec-20', calculate_auc_Node2Vec(g, poss_set, neg_set, 20))
    #add_auc(auc, 'Node2Vec-128', calculate_auc_Node2Vec(g, poss_set, neg_set, 128))
    
    add_auc(auc, "PA", calculate_auc_PA(g, poss_set, neg_set))
    add_auc(auc, "CN", calculate_auc_CN(g, poss_set, neg_set))
    add_auc(auc, "Adamic-Adar", calculate_auc_Adamic_Adar(g, poss_set, neg_set))
    
    add_auc(auc, "NMF-10", 
        calculate_auc_NMF(10, train_set, nodes, poss_set, neg_set, directed))
    add_auc(auc, "svds-30", 
        calculate_auc_SVDS(30, train_set, nodes, poss_set, neg_set, directed))
    add_auc(auc, "NMF-30", 
        calculate_auc_NMF(30, train_set, nodes, poss_set, neg_set, directed))
    add_auc(auc, "svds-10", 
        calculate_auc_SVDS(10, train_set, nodes, poss_set, neg_set, directed))

    add_auc(auc, "sfdp-default", calculate_auc_default(g, max_iter, poss_set, neg_set, p, directed))    

    if bipartite:
        is_bi, part = graph_tool.topology.is_bipartite(g, partition=True)
        if is_bi:
            groups = g.new_vertex_property("int")
            for u in g.vertices():
                groups[u] = int(part[u])
            left = "repulse-aliens"
            right = "repulse-aliens"
            pos_bip = sfdp_layout(g, groups=groups, verbose=verbose, 
                                      bipartite=True, bipartite_method=[left, right], gap=gap)
            features = tools.TopologicalFeatures(g, pos_bip, gap=gap)
            X, Y = tools.make_dataset(poss_set, neg_set, [features.dist])
            add_auc(auc, "sfdp-bipartite", roc_auc_score(Y, X))  
    return 
                       
        
    print('g_di')
    g_di = Graph(directed=False)
    g_di.add_vertex(2*max(nodes) + 2)
    for edge in train_set:
        if edge not in poss_set:
            u, w = map(int, edge.split())
            if g.vertex(u).out_degree() / g.vertex(w).out_degree() < 0.5:
                g_di.add_edge(g_di.vertex(2*u + 1), g_di.vertex(2*w))
            elif g.vertex(u).out_degree() / g.vertex(w).out_degree() > 2:
                g_di.add_edge(g_di.vertex(2*w + 1), g_di.vertex(2*u))
            else:
                g_di.add_edge(g_di.vertex(2*u + 1), g_di.vertex(2*w))
                g_di.add_edge(g_di.vertex(2*w + 1), g_di.vertex(2*u))
                    
    
    
    #auc["sfdp-directed"].append(calculate_auc_directed(g_di, verbose, gap, 
    #                                                   convert_set(poss_set, g), 
    #                                                   convert_set(neg_set, g)
    #                                                  ))
    
    if directed:
        if "sfdp-directed" not in auc:
            auc["sfdp-directed"] = []
        auc["sfdp-directed"].append(calculate_auc_directed(g, verbose, gap, poss_set, neg_set))
   
    return auc


def cross_validation(file, N, k, gap=0, verbose=False, directed=False, bipartite=False, max_iter=0, 
                     comment='', p=2, sparse=False):
    auc = {}
    
    train_set, nodes, poss_set, neg_set = None, None, None, None
    poss_set_2, neg_set_2 =  None, None
    
    for i in range(k):
        train_set, nodes, poss_set, neg_set = None, None, None, None
        
        if bipartite:
            train_set, nodes, poss_set, neg_set = tools.sample_bipartite(file, N)
        else:
            train_set, nodes, poss_set, neg_set = tools.sample_structural(file, N, directed=directed, sparse=sparse)
                
        print('calculate_auc', len(train_set), len(nodes), len(poss_set), len(neg_set))
        calculate_auc(train_set, nodes, poss_set, neg_set, auc, gap, verbose, directed, bipartite, max_iter, p)
    
    FIN = open('cross_validation', 'a')
    
    METADATA = {
        'file': file,
        'N': N,
        'k': k,
        'gap': gap,
        'verbose': verbose,
        'directed':directed,
        'bipartite':bipartite,
        'max_iter': max_iter,
        'p': p,
        'sparse': sparse,
        'comment': comment,
        'auc': auc
    }
    
    FIN.write(json.dumps(METADATA, sort_keys=True, indent=2, separators=(',', ': ')) + '\n')
    
    for x in auc:
        s = x + ' ' +  ": %0.6f (+/- %0.6f)" % (np.array(auc[x]).mean(), np.array(auc[x]).std() * 2)
        print s
        
    FIN.close()
    