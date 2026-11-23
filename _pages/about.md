---
layout: about
title: about
permalink: /
subtitle: PhD student, Representation Learning Lab, <a href="https://iisc.ac.in/">Indian Institute of Science</a>, Bangalore.

# The profile block controls the photo and the contact card on the right.
# The image is read from assets/img/, so drop the headshot there under this name.
profile:
  align: right
  image: prof_pic.jpg
  image_circular: false
  more_info: >
    <p>Indian Institute of Science</p>
    <p>Bangalore 560012, India</p>
    <p><a href="mailto:vanditpatel.iisc@gmail.com">vanditpatel.iisc@gmail.com</a></p>

selected_papers: false # No publications yet, so the front page keeps this off.
social: true # Renders the icon row driven by the social keys in _config.yml.

announcements:
  enabled: true
  scrollable: true
  limit: 5

latest_posts:
  enabled: true
  scrollable: true
  limit: 3
---

**I am interested in what a model actually represents, and in whether we can work with that representation directly: predict in it, generate it in parallel, and read or edit it in the weights.**

I am interested in what a model actually represents, and in whether we can work with that representation directly: predict in it, generate it in parallel, and read or edit it in the weights.

I am a PhD student in the Representation Learning Lab at IISc, advised by Prof. Prathosh A.P. My work follows three threads. I think of them as one question asked at different levels of a model.

The first is next-concept modelling. An autoregressive language model factorises text one token at a time, $$p(x) = \prod_t p(x_t \mid x_{<t})$$, even though the ideas it expresses span many tokens. I am interested in models that predict in a space of concepts instead, $$p(c_k \mid c_{<k})$$, decoding tokens only afterwards. I want to understand what that change buys in coherence, planning and sample efficiency, and what it costs.

The second is diffusion language models, which drop left-to-right order altogether and produce a whole sequence by iteratively denoising it. Steering them fits the controlled-process view of generation: guidance and reward fine-tuning tilt the pretrained measure toward $$\pi^\star \propto p_{\text{pre}}\exp(r/\beta)$$. A finite-step learned sampler can only approximate that tilt, and I want to know what the approximation costs when the space is discrete.

The third is weight-space programming, which treats a trained network's parameters as an object to read, generate and edit. If weights can be read, then alignment can be audited where it actually lives, in the weights, rather than only through a model's behaviour, and possibly repaired there too. My first experiment in this direction asked whether a language model can work out what a network was trained to do from its raw weights alone. Mostly it can't, and the ways it fails are the interesting part: read the write-up.

I took a roundabout route here. I trained as an electrical engineer at Nirma University, where my undergraduate project tried to predict faults in induction machines from their frequency response. For my master's at IISc, in the Machine Learning Lab under Prof. Chiranjib Bhattacharyya, I recast optimally timed intervention as an optimal stopping problem on a hazard process, so that "when do we act?" has a principled answer rather than a tuned threshold. I then spent two years as an ML Engineer at Stryker AI Research, working on agentic AI, post-training and generative modelling under tight data constraints. Building systems whose failures I could see but not bound is a large part of why I came back to theory.

If any of this overlaps with what you are working on, please write. I am always glad to talk, and open to collaboration.
