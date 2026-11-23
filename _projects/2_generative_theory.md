---
layout: page
title: control and inference in diffusion models
description: Ongoing PhD work on steering generative processes, and on what the objectives certify.
img: assets/img/projects/diffusion.jpg
importance: 1
category: research
---

Ongoing work in the Representation Learning Lab at IISc.

Guidance, reward fine-tuning and inference-time search all reshape the measure a diffusion model samples from. Written as a KL-regularised control problem, the optimal policy is an exponential tilt of the pretrained measure, $$\pi^\star \propto p_{\text{pre}}\exp(r/\beta)$$ — clean in closed form, and not something a finite-horizon learned controller can realise exactly.

The questions I am working on concern that gap. How much bias does an approximate tilt introduce, and can it be bounded in terms of quantities we can actually estimate during training? What happens to mode coverage under repeated steering? And when a variational bound is applied along a trajectory rather than to a single observation, how do the per-step gaps compose?

This connects back to older ground I keep returning to: the failure modes of variational objectives, the geometry of push-forward measures, and why integral probability metrics were introduced in the first place. Notes on these are in the [blog]({{ '/blog/' | relative_url }}).
