- The simple algorithm improvment is to make global repulsive forses only between groups. So, the nodes from the same group will not interact with each other. It will help to consentrate more nodes around the same interest.

Movielens experiment have shown the following growth in accuracy.
The speed is higher than simple sfdp.

5%                                           (movielens)  (frwiki) , (condmat)        (crime)
sfdp_layout(g)                                : 0.841    , ------- ,  0.528 (3 iter) , 0.569
sfdp_layout(g, groups=groups)                 : 0.814    , ------- ,  -----          , 0.370
sfdp_layout(g, groups=groups, bipartite=True) : 0.887    , 0.843   ,  0.627,         , 0.452
PA                                            : 0.902    , 0.844   ,  0.590,         , 0.766
NMF(10)                                       : 0.925    , 0.723   ,  0.579,         , 0.552
SVDS(30)                                      : 0.943    , 0.678   ,  0.668,         , 0.568





