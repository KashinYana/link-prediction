- The simple algorithm improvment is to make global repulsive forses only between groups. So, the nodes from the same group will not interact with each other. It will help to consentrate more nodes around the same interest.

Movielens experiment have shown the following growth in accuracy.

5%                                           (movielens)  (github)
sfdp_layout(g)                                : 0.841    ,      
sfdp_layout(g, groups=groups)                 : 0.814    ,
sfdp_layout(g, groups=groups, bipartite=True) : 0.887    ,
PA                                            : 0.902    ,
NMF(10)                                       : 0.925    ,
SVDS(30)                                      : 0.943    , 




