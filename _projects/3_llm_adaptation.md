---
layout: page
title: adapting language models under data constraints
description: Industry research at Stryker on parameter-efficient adaptation, retrieval coupling, and synthetic data.
img: assets/img/projects/llm.jpg
importance: 4
category: industry
---

Two years of applied research at Stryker AI Research, all of it shaped by the same constraint: domain-specific systems that had to work with far less labelled data than the methods were designed for.

**Parameter-efficient adaptation.** Studied low-rank fine-tuning of large language models for domain adaptation, and the trade-off it forces between the expressive power a rank constraint permits and the generalisation error it buys back.

**Retrieval–generation coupling.** Investigated how retrieval quality propagates through agentic pipelines. Errors introduced at retrieval do not stay local; framing this as a bias–variance question made the sample-efficiency argument for improving retrieval before scaling generation.

**Ranking under distribution shift.** Built a resume-matching framework and analysed the stability of learning-to-rank when the deployment distribution drifts from the training one.

**Synthetic data for low-data regimes.** Designed generative pipelines to augment scarce data, focusing on mode coverage and divergence minimisation — the practical version of the theoretical question of whether a generative model has actually covered the support it claims to.

This work also produced a patent on hand and instrument detection for automatic instrument configuration and control.
