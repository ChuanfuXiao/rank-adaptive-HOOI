# Rank-adaptive HOOI
A novel algorithm for solving the fixed-accuracy low multilinear-rank approximation of tensors, i.e., 
$$\min\limits_{\mathcal{B}}\mu\text{rank}(\mathcal{B}),\  \text{s.t.}\  ||\mathcal{A}-\mathcal{B}||\leq\varepsilon||\mathcal{A}||$$
where $\varepsilon\in(0,1)$ is a predetermined accuracy requirement.

Classical methods for solving the fixed-accuracy low multilinear-rank approximation problem include:
+ Truncated HOSVD
+ Sequentially truncated HOSVD
+ Greedy HOSVD
