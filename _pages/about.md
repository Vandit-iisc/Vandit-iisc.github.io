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

**I study what our training objectives actually certify about the models they produce, and where that guarantee quietly stops holding.**

I am a PhD student in the Representation Learning Lab at IISc, advised by [Prof. Prathosh A.P.](https://sites.google.com/view/prathosh) At the moment that mostly means diffusion models seen as controlled stochastic processes. Guidance, reward fine-tuning and inference-time search all tilt the pretrained measure toward $$\pi^\star \propto p_{\text{pre}}\exp(r/\beta)$$. A finite-horizon learned controller can only approximate that tilt. I want to know what the approximation costs, and whether the steered sampler still covers the modes it started with.

I took a roundabout route here. I trained as an electrical engineer at Nirma University, and my undergraduate project modelled faults in induction machines and tried to predict them from the machine's frequency response. The question underneath it was *when will this fail, and what should we do before it does?*, and it came with me to IISc. For my master's, in the Machine Learning Lab under [Prof. Chiranjib Bhattacharyya](https://www.csa.iisc.ac.in/~chiru/), I recast optimally timed intervention as an optimal stopping problem on a hazard process. That makes "when do we act?" a question the problem itself answers, rather than a threshold tuned on a validation set.

After that I spent two years as an ML Engineer at **Stryker AI Research**, working on agentic AI, post-training and generative modelling. The systems had to be adapted to specialised domains with far less labelled data than the methods assume. Building systems whose failures I could see but not bound is a large part of why I came back to theory.

Apart from the main line of work, I like small, slightly odd experiments. The latest one asked whether a language model can work out what a neural network was trained to do from its raw weights alone. Mostly it can't, and the ways it fails are the interesting part: [read the write-up]({{ '/blog/2026/11/recovering-task-from-weights/' | relative_url }}).

I am also drawn to bandit algorithms. For me they are the cleanest setting for pricing the cost of not knowing, and I keep finding them inside generative pipelines, where the arms are prompts, candidate samples or intervention times.

If any of this overlaps with what you are working on, please write. I am always glad to talk, and open to collaboration.
