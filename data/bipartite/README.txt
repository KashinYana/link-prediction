- The simple algorithm improvment is to make global repulsive forses only between groups. So, the nodes from the same group will not interact with each other. It will help to consentrate more nodes around the same interest.

Movielens experiment have shown the following growth in accuracy.
The speed is higher than simple sfdp.

5%                                           (movielens)  (frwiki) , (condmat)        (crime)
sfdp_layout(g)                                : 0.841    , 0.757   ,  0.765,         , 0.569
sfdp_layout(g, groups=groups)                 : 0.814    , 0.724   ,  0.606,         , 0.370
sfdp_layout(g, groups=groups, bipartite=True) : 0.887    , 0.847   ,  0.646,         , 0.452
PA                                            : 0.902    , 0.845   ,  0.597,         , 0.766
NMF(10)                                       : 0.925    , 0.749   ,  0.567,         , 0.552
SVDS(30)                                      : 0.943    , 0.690   ,  0.671,         , 0.568


# Cross validation test 5%

## Crime:
svds-30 : 0.55 (+/- 0.05)
NMF-10 : 0.47 (+/- 0.05)
sfdp-bipartite-simple : 0.32 (+/- 0.07)
PA : 0.39 (+/- 0.09)
sfdp-default : 0.61 (+/- 0.12)


## Movielens:
svds-30 : 0.94 (+/- 0.00)
NMF-10 : 0.93 (+/- 0.00)
sfdp-bipartite-simple : 0.89 (+/- 0.00)
PA : 0.90 (+/- 0.00)
sfdp-default : 0.84 (+/- 0.00)


## Condmat:
svds-30 : 0.66 (+/- 0.02)
NMF-10 : 0.57 (+/- 0.01)
sfdp-bipartite-simple : 0.66 (+/- 0.01)
PA : 0.59 (+/- 0.01)
sfdp-default : 0.76 (+/- 0.01)

## Frwiki:
svds-30 : 0.70 (+/- 0.01)
NMF-10 : 0.74 (+/- 0.01)
sfdp-bipartite-simple : 0.85 (+/- 0.01)
PA : 0.85 (+/- 0.01)
sfdp-default : 0.76 (+/- 0.02)





