<div align="center">

# Replication of [IJCAI 2024] Learning Fair Representations for Recommendation via Information Bottleneck Principle

<p>
Copy of the implementation of "Learning Fair Representations for Recommendation via Information Bottleneck Principle", IJCAI 2024.
</p>
<img src='fig.png' width='80%'>
</div>

This repository is part of a university course project on paper reproducibility. The results achieved by its authors are noted on [google sheets](https://docs.google.com/spreadsheets/d/11S1aDUdlzUSGnC6tsRgjBvusY3FyirYNT841ihV-qnI/edit?usp=sharing).


## Rest of the original description

In this paper, we research fairness-aware recommender systems from the information theory perspective. Motivated by the information bottleneck principle, we propose a novel model-agnostic fair representation method FairIB to eliminate the sensitive information from the learned representations. Specifically, FairIB maximizes the mutual information between learned representations and observed interactions, meanwhile minimizing it between representations and user sensitive attributes. To achieve this goal, we introduce HSIC-based bottleneck to recommender systems, and 
applied to both the user and sub-graph sides. Extensive experiments on two real-world datasets demonstrated FairIB is effective in efficient recommendation accuracy-fairness trade-off, either in single or multiple sensitive scenarios.

Run
--------------
- Training FairIB_BPR on MovieLens: 

```shell
python fairib_bpr_movie.py
```


- Training FairiB_LightGCN on MovieLens: 

```shell
python fairib_gcn_movie.py
```
Author contact:
--------------
Email: jsxie.hfut@gmail.com
