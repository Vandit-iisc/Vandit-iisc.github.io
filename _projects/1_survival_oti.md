---
layout: page
title: optimally timed interventions
description: Survival analysis for critical events, posed as optimal stopping on a hazard process.
img: assets/img/projects/survival.jpg
importance: 3
category: research
related_publications: false
---

Master's research at IISc under Prof. Chiranjib Bhattacharyya, on deciding *when* to intervene before a critical event — an equipment failure, a security breach, a medical emergency.

The core contribution is a reduction. Rather than treating intervention timing as a thresholding heuristic on a predicted risk score, the problem is posed as an **optimal stopping problem on a hazard rate process**. This establishes the equivalence formally, which means the resulting policy inherits the guarantees of optimal stopping theory instead of resting on a tuned cutoff.

On the modelling side, the work develops a **DDRSA** architecture that estimates the conditional hazard function directly. Modelling long monitoring histories efficiently required a Transformer with ProbSparse attention, which keeps the cost of attending over a long sensor trace manageable while preserving the temporal structure the hazard estimate depends on.

Applied to the NASA turbofan degradation benchmark, the framework produces intervention policies that are sensitive to how the hazard is *conditioned* — an issue that turns out to matter more than the choice of architecture, and one I am still thinking about.
