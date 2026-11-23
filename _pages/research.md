---
layout: page
title: research
permalink: /research/
description: What I work on, and the questions underneath it.
nav: true
nav_order: 2
---

I am interested in the theory of generative models and sequential decision making, and in the places where the two meet. Broadly, I want to understand what our training objectives actually certify about a learned model, and where the guarantee quietly stops holding.

---

## Control and reinforcement learning in diffusion models

A diffusion model defines a reverse-time SDE whose drift is set by the learned score,

$$
\mathrm{d}x_t = \Big[ f(x_t, t) - g(t)^2 \nabla_x \log p_t(x_t) \Big]\mathrm{d}t + g(t)\,\mathrm{d}\bar{w}_t .
$$

Once we start steering that process — classifier guidance, reward fine-tuning, inference-time search — we are no longer sampling from $$p_{\text{data}}$$; we are solving a stochastic optimal control problem whose value function reweights the path measure. Writing the fine-tuning objective in KL-regularised form,

$$
\max_{\pi}\ \mathbb{E}_{x \sim \pi}\big[r(x)\big] - \beta\, D_{\mathrm{KL}}\!\left(\pi \,\|\, p_{\text{pre}}\right),
$$

makes the trade-off explicit: the optimal $$\pi^\star \propto p_{\text{pre}} \exp(r/\beta)$$ is an exponential tilt of the pretrained measure, and the interesting questions are about what happens when we can only approximate that tilt with a finite-horizon, finitely-parameterised controller. I am interested in the bias this introduces, and in whether the resulting sampler retains any coverage guarantee over the modes of the base model.

## Bandit algorithms and their applications

Bandits are the cleanest setting I know for reasoning about the cost of not knowing. The regret decomposition

$$
R_T = \sum_{a \neq a^\star} \Delta_a \, \mathbb{E}[N_a(T)]
$$

reduces algorithm design to a question about how many times a suboptimal arm must be pulled before it can be ruled out, which in turn is an information-theoretic question about distinguishing distributions. I am interested in this both on its own terms and as a lens on adaptive data collection inside generative pipelines, where the "arms" are candidate samples, prompts, or intervention times.

## Foundations of generative modelling

Most of what we deploy is a surrogate. The ELBO,

$$
\log p_\theta(x) = \mathcal{L}(\theta,\phi;x) + D_{\mathrm{KL}}\!\left(q_\phi(z \mid x) \,\|\, p_\theta(z \mid x)\right),
$$

is exact only in the sense that the gap is a KL divergence we never measure. Adversarial objectives replace it with a different surrogate — an $$f$$-divergence or, for Wasserstein GANs, an integral probability metric chosen precisely because divergences degenerate when the model and data measures are mutually singular, as they generically are on a low-dimensional manifold. I care about characterising these gaps: when posterior collapse is a property of the objective rather than of the optimiser, what the push-forward geometry implies about mode coverage, and how bias compounds when an ELBO is applied along a trajectory rather than to a single observation.

## Survival analysis and optimal stopping

My master's work formulated *optimally timed intervention* as an optimal stopping problem on a hazard process. Given a conditional hazard $$\lambda(t \mid \mathcal{F}_t)$$, the survival function is

$$
S(t) = \exp\left(-\int_0^t \lambda(u \mid \mathcal{F}_u)\,\mathrm{d}u\right),
$$

and the intervention policy becomes a stopping time $$\tau$$ maximising an expected utility that trades the cost of acting early against the risk of the event occurring first. Establishing that equivalence is what makes the problem tractable: it turns "when should we act?" into a question with a principled answer rather than a tuned threshold.

## Reading models from their weights

A smaller, more exploratory thread asks the inverse question: given a trained network's parameters $$\theta$$ and nothing else, how much can be recovered about the task that produced them? For a ReLU network the map $$x \mapsto f_\theta(x)$$ is piecewise linear, so the effective linear model $$W_{\text{eff}}(x)$$ depends on which units $$x$$ activates. Any global summary of $$\theta$$ is an average over activation patterns. In a [first experiment]({{ '/blog/2026/11/recovering-task-from-weights/' | relative_url }}), language-model agents recovered the dataset and the dead units but not the decision rule. The broader point is that an explanation read from weights is a hypothesis until it is tested against activations on real inputs.

---

Details of individual projects are on the [projects]({{ '/projects/' | relative_url }}) page, and longer derivations live in [notes]({{ '/blog/' | relative_url }}).
