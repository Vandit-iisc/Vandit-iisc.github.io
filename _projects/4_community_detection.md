---
layout: page
title: community structure in real networks
description: Spectral and modularity-based community detection on social and transaction graphs.
img: assets/img/projects/networks.jpg
importance: 5
category: coursework
---

Implemented and compared community detection algorithms — Karger's min-cut, Girvan–Newman, and Louvain — on Facebook social graphs and Bitcoin transaction networks.

The interesting part was less the implementation than the comparison. Girvan–Newman's repeated edge-betweenness recomputation makes it the most interpretable and the least scalable of the three, while Louvain's greedy modularity optimisation is fast enough for the transaction graph but offers no guarantee that the partition it converges to is near-optimal. Evaluating modularity scores alongside runtime made the trade-off concrete, and the resulting partitions differed enough between methods to be worth taking seriously as a caution about reading community structure off any single algorithm.
