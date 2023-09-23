# SSE
### Semi-supervised clustering via structural entropy with different constraints.
![image](framework.png)
Overview of SSE. (I) Two graphs G and G' are constructed from input data and constraints, respectively. (II) Semi-supervised partitioning clustering is performed through two opertors merging and moving. (III) Semi-supervised hierarchical clustering is performed through two operators stretching and compressing.

# Installation
Install the required packages listed in the file ```requirement.txt```. The code is tested on Python 3.10.0.

# Usage
In the root directory of this project:
```
python main.py [-h][--method METHOD][--dataset DATASET]
               [--constraint_ratio RATIO][--constraint_weight WEIGHT]
               [--sigmasq SIGMASQ][--exp_repeats REPEATS]
               [--knn_constant KNN_CONSTANT][--hie_knn_k HIE_KNN_K]
```

example: ```python main.py --method SSE_hierarchical --dataset wine --constraint_ratio 0.2```
```
required arguments:
  --method METHOD    running different components of SSE. Choices are SSE_partitioning_pairwise, SSE_hierarchical, and so on.
  --dataset DATASET    dataset to run. They should be stored in directory ./datasets.
  --constraint_ratio   constraint ratio. Recommend setting 0.2 for pairwise constraints and 0.1 for label constraints.
```
```
optional arguments:
  --constraint_weight     weight for penalty term. (default 2).
  --sigmasq SIGMASQ       square of Gaussian kernel band width, i.e., sigma^2.
  --exp_repeats REPEATS   number of experiment repeats. (default 10).
  --knn_constant          a constant for graph construction in partitioning clustering. (default 20).
  --hie_knn_k             a constant for graph construction in hierarchical clustering. (default 5).
```
