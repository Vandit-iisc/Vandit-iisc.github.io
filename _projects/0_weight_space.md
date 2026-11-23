---
layout: page
title: reading a network from its weights
description: Can a language model recover what a classifier was trained to do, given only its parameters?
importance: 2
category: research
related_publications: false
---

A small experiment with Prayalankar and Prof. Prathosh A.P., inspired by the *Weight-Space Programming* work from MIT CSAIL.

The network is a one-hidden-layer MLP with 100,609 parameters. It maps an MNIST digit to a single bit, and was trained to separate $$\{0, 2, 3, 5, 7\}$$ from the other digits. Language-model agents were given the weights and nothing else: they could not run the network on inputs. Their job was to name the rule.

Mostly they could not. A frontier model with one labelled example proposed even versus odd, which scores below a random split. Tool-using agents working blind settled confidently on "zero versus everything else", with two independent analyses telling the same wrong story. A hint that each class has exactly five digits removes only $$\log_2(1022/252) \approx 2$$ bits of ambiguity, yet it moved every agent to within one or two digits of the truth. Hints act as data, and a report that does not say what the model was told is hard to interpret.

The more durable lesson is about why weights resist reading. Adam-trained weights are initialisation plus error-weighted sums of training images, so what they store are *differences* between look-alike digits. The network is piecewise linear, which means the decision depends on which units each input switches on, and no single global map captures it. Weights reveal structure (the dataset, dead units, a default class) but not the rule. Claims about what the weights mean stay hypotheses until they are checked against activations on real inputs.

[Full write-up]({{ '/blog/2026/11/recovering-task-from-weights/' | relative_url }}) · [code and data](https://github.com/Vandit-iisc/Recovering-Neural-Network-Task)
