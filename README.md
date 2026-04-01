# Feature-weighted-FKNN-regression-using-fuzzy-mutual-information-and-Lukasiewicz-similarity

This is a new method to the family of fuzzy k-nearest neighbor (FKNN) regression based on the use of feature weights and Minkowski distance. This method is called feature-weighted Minkowski distance-based fuzzy $k$-nearest neighbor regression method (FWMD-FKNNreg). 

### Matlab functions:

The main functions include the the FWMD-FKNNreg algorithm (`mink_weighted_fknnreg.m`) and feature weights compuation (`generate_feature_weights.m`). In addition to those files, an example (`example_run.m`) of the use of FWMD-FKNNreg is also presented. `FHjoin.m`,`MC.m`, `simL`, and `simR` are needed to compute feature weights based on relevance, redundancy, and dependency.

Reference: [Kumbure, M. M., Luukka, P., Collan, M.: Feature-weighted FKNN regression using fuzzy mutual information and Łukasiewicz similarity. In: Proceeding of the IEEE International Conference on Fuzzy Systems (FUZZ-IEEE-2026)]

Created by Mahinda Mailagaha Kumbure & Pasi Luukka 03/2026
