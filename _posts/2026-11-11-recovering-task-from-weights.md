---
layout: post
title: "Can an LLM Read a Neural Network? Recovering a Classifier's Task from Its Weights"
description: "We handed GPT-6 Astra the raw weights of a small MLP and asked what it was trained to do. A single call guessed parity. A team of agents found the real rule."
authors: Vandit Patel, Prayalankar, Prathosh AP
---

Suppose I give you a file of 100,609 floating-point numbers and tell you they are the weights of a small network that takes a 28×28 grayscale image and outputs a single bit. Nothing else: no training data, no labels, no description of the task. Could you tell me what the network was trained to do?

That's the question this post is about, with a large language model doing the reading instead of a person. The short version: a single long-reasoning call to GPT-6 Astra figured out that the network looks at handwritten digits but guessed the wrong rule (even vs. odd, at 42% confidence). A hierarchy of agents working on the same weights produced several close candidates, and one branch got the exact rule.

## Where the question came from

The starting point was the experiments around *Sparks of Weight Space Programming*, which look at whether LLMs can interpret and edit neural network parameters directly. One experiment there shows an LLM the weights of a game-playing policy network, along with a description of its architecture, input channels and actions, and asks which game it plays. The authors report correct identifications for four selected policies, each from a single response in a fresh conversation.

The idea behind weight-space programming is that parameters become something an LLM can inspect, describe, generate or modify. We only look at the *interpretive* direction here: given a trained parameter set, can the model work out what task the computation implements? (Editing the weights and showing changed behaviour would be a separate intervention experiment.)

## The setup

The network is a plain one-hidden-layer MLP:

<figure>
  <img src="{{ '/assets/img/fig1_pipeline.png' | relative_url }}" alt="Pipeline: 28x28 image, flattened to 784, 128 ReLU hidden units, one raw logit, thresholded at zero">
  <figcaption>Figure 1. The classifier we asked the LLM to interpret.</figcaption>
</figure>

Writing an image as $$X \in [0,255]^{28\times 28}$$ and its normalised, flattened form as $$x = \mathrm{vec}(X/255) \in \mathbb{R}^{784}$$, the model computes

$$
h(x) = \mathrm{ReLU}(W_1 x + b_1), \qquad z(x) = W_2\, h(x) + b_2, \qquad \hat y(x) = \mathbb{1}\{z(x) > 0\},
$$

with $$W_1 \in \mathbb{R}^{128\times 784}$$, $$b_1 \in \mathbb{R}^{128}$$, $$W_2 \in \mathbb{R}^{1\times 128}$$ and $$b_2 \in \mathbb{R}$$. It was trained with `BCEWithLogitsLoss`, so $$z$$ is a raw logit and $$\sigma(z) > 0.5$$ means class 1. Flattened index $$i$$ maps to pixel row $$\lfloor i/28 \rfloor$$ and column $$i \bmod 28$$, which matters as soon as you start reading rows of $$W_1$$ as images.

Formally, we want an inference procedure $$\mathcal{A}$$ that produces a semantic hypothesis

$$
\hat H = \mathcal{A}(\theta, M, X, y),
$$

from the parameters $$\theta = \{W_1, b_1, W_2, b_2\}$$, architectural metadata $$M$$, and optionally one labelled example $$(X, y)$$. For digit classification the exact target is a map $$g : \{0,\dots,9\} \to \{0,1\}$$. The network we used was trained on

$$
g(d) = \begin{cases} 1 & d \in \{0, 2, 3, 5, 7\} \\ 0 & d \in \{1, 4, 6, 8, 9\} \end{cases}
$$

that is, primes plus zero against everything else. (Calling it "prime vs. non-prime" would leave zero's side undecided, so we state the partition explicitly.) We score a guess two ways. **Domain recovery** asks whether the model realised these are handwritten digits. **Exact partition recovery** asks whether it put all ten digits on the right side, with the label orientation preserved. Partial credit is counted as digit assignments out of 10. That is a comparison between symbolic sets, not image-level accuracy.

## Attempt 1: one long call

We sent GPT-6 Astra one request through the Responses API with high reasoning effort and no tools (`tools = []`, `tool_choice = none`). It contained the full-precision weights, the normalisation and flattening conventions, the activation and loss, and one randomly chosen labelled image from our 50-image CSV (row 44, picked with a recorded seed).

One honest detail: the *first* request failed. It had a 4,096-token output budget, spent all of it on reasoning, and stopped with `max_output_tokens` before giving any answer. The second request raised the budget to 32,768 tokens and asked for a shorter visible answer. It used **669,642 input tokens** and **14,546 output tokens**, 14,224 of them reasoning, and finished normally. When I say "single-call baseline" below, I mean this completed second request.

Astra described the example image as a handwritten zero and proposed

$$
\text{class 1} = \{0,2,4,6,8\}, \qquad \text{class 0} = \{1,3,5,7,9\},
$$

at confidence 0.42, calling parity tentative and noting that other partitions were possible.

This was not a blind guess. The answer cited specific parameters, for example $$W_1[2,349] \approx -0.635$$ paired with $$W_2[0,2] \approx +0.478$$, and $$W_1[12,462] \approx +0.330$$ paired with $$W_2[0,12] \approx -0.914$$, and these values match the file. The model really was reading the weights. It then reached for a familiar rule, and parity agrees with the truth on only 0, 1, 2 and 9: **4/10 digits, no exact recovery.**

## Attempt 2: a hierarchy of agents

Next we split the problem up. Four main agents (A, B, C, D) each ran three subagents, giving 12 subagents in total. The subagents analysed the weights, mostly by reading rows of $$W_1$$ as spatial templates, and each main agent combined its branch's findings into a ranked list of candidate partitions.

<figure>
  <img src="{{ '/assets/img/fig2_partitions.png' | relative_url }}" alt="Grid of digits 0-9 coloured by class for the ground truth, the single-call answer, and each agent's top pick; mismatches outlined">
  <figcaption>Figure 2. Every proposed partition, digit by digit. Dashed outlines mark disagreements with the ground truth. Agent C's top pick is exact; A and D each get 8/10.</figcaption>
</figure>

Agent C's preferred hypothesis was exactly $$\{0,2,3,5,7\}$$ vs. $$\{1,4,6,8,9\}$$. A and D each placed eight of ten digits correctly (A's top pick was also C's second choice), and B's top pick got six. One caveat on how to read this: *one branch* found the exact answer. If we had simply ranked all candidates together, by mean score for example, B's partition would have come out on top. Recovery by an individual branch and the aggregate ranking are different things, and the second is less impressive.

Note also what every candidate agrees on: 2 and 3 always go to class 1, and 1 and 4 always go to class 0. The disagreements are concentrated on 5, 6, 7 and 8.

## How do you read a task off weights?

The agents' main tool was to treat each hidden unit as a spatial detector. For hidden unit $$j$$, let $$w_j = W_1[j,:]^\top \in \mathbb{R}^{784}$$, $$b_j = b_1[j]$$, and $$v_j = W_2[0,j]$$. Its pre-activation and contribution to the logit are

$$
a_j(x) = w_j^\top x + b_j, \qquad c_j(x) = v_j \max\big(0,\, a_j(x)\big), \qquad z(x) = \sum_{j=0}^{127} c_j(x) + b_2 .
$$

Reshape $$w_j$$ to 28×28 and you get a template: where it is positive, ink excites the unit; where it is negative, ink suppresses it. The sign of $$v_j$$ then says whether that unit pushes toward class 1 or class 0. There is also an indirect route. If a class-0 unit ($$v_j<0$$) has negative weights in some region, ink there *switches it off*, which raises the logit through disinhibition.

**A default answer for blank input.** Several strongly class-0 units had small positive biases (about $$+0.10$$ to $$+0.21$$), mostly negative input weights, and the output bias was near $$-0.34$$. That points to a design where a blank image defaults toward class 0, and ink in the right places turns that suppression off. The blank-image logit is

$$
z(\mathbf{0}) = \sum_{j=0}^{127} v_j \max(0, b_j) + b_2 ,
$$

so the default depends on the whole sum, not only on the sign of $$b_2$$.

**Where the evidence sits.** The agents pointed to four regions of the image:

<figure>
  <img src="{{ '/assets/img/fig3_regions.png' | relative_url }}" alt="28 by 28 pixel grid with two blue regions on the left favouring class 1 and two orange regions in the centre and upper right favouring class 0">
  <figcaption>Figure 3. Image regions the agents flagged. Blue regions carry evidence for class 1 and orange regions for class 0. Rows and columns follow the flattening convention above.</figcaption>
</figure>

The bottom-left region R1 (rows 18–24, columns 1–5) was the clearest class-1 signal, with roughly 16 units of aggregate weight favouring ink against about 1 opposing it. That matches the base of a 2, the bottom tails of 3 and 5, and rounded contours generally. The upper-left tips R2 match where the arcs of 2 and 3 begin.

On the class-0 side, the central vertical band R3 is where a 1 is drawn and where the stems of 4 and 9 sit. The upper-right strip R4 is where the top bar of a 7 (or 5) ends:

<figure>
  <img src="{{ '/assets/img/fig4_support.png' | relative_url }}" alt="Bar chart: class-1 units oppose ink in the central band and upper right much more than they favour it; class-0 units favour central ink">
  <figcaption>Figure 4. Aggregate weight favouring vs. opposing ink, as reported by the agents. Class-1 units are strongly suppressed by ink in the centre and upper right, while class-0 units welcome central ink.</figcaption>
</figure>

**Why the agents disagreed.** The hard digits are those whose strokes cross regions with opposite meanings. A 5 has an upper-left stroke and a bottom-left tail (class 1), but its middle stroke falls in class-0 territory, and the branches' scores for 5 ranged from 0.08 to 0.75. A 7 is permitted by some units (39, 27) while others (73, 23, 96, 33) specifically penalise the end of its top bar. An 8's outer loops look like class 1, but units 36, 69 and 17 appear to separate 3 from 8. Different subsets of features favour different partitions, which is exactly the pattern in Figure 2.

> This spatial story is a *proposed* explanation, not a demonstrated one. The "favouring/opposing" totals summarise raw weights, and they are not logit contributions because they ignore ReLU gating, biases and $$v_j$$. Turning this into a causal account requires measuring $$c_j(x)$$ on real digit examples and running targeted pixel and unit interventions.

## Were the agents smarter, or did they just do more?

Distributing the work plausibly helps. Different branches can look at different parts of the weights, keep competing hypotheses alive, and criticise each other instead of committing early to a familiar rule like parity. The spread of candidates in Figure 2 fits that picture.

But this experiment doesn't isolate *organisation* as the cause. The agentic run may have used more tokens, tool calls, numerical scripts, different prompts, wider file access or more examples. Each of those could matter on its own. The fair framing is that we observed two inference configurations, not a controlled comparison. Making the causal claim requires a matched-access, matched-compute comparison, and that is the obvious next experiment.

Two smaller caveats. The agents' claims that 67 pixel columns kept their initial weights and that norms grew about 5× need the initial checkpoint to verify. Blank-pixel patterns alone don't uniquely identify MNIST, and they don't rule out transformations of it.

## Takeaways

1. **Domain recovery is easy-ish, rule recovery is hard.** A single call correctly read "handwritten digits" from raw weights and still got 6 of 10 digits wrong on the actual rule.
2. **Familiar priors compete with evidence.** Parity is the first binary rule over digits most people (and models) think of, and the single call settled on it despite citing real weights.
3. **Decomposition surfaced the right answer.** One of four agent branches recovered the exact partition. Aggregating across branches would not have chosen it.
4. **The next step is controls.** Match information and compute between single-call and agentic runs, and verify the spatial story with activation measurements and interventions.

## Reproducibility

We keep the model weights, the selected example, prompt hashes, and the direct API requests and responses. A fully reproducible agentic comparison would also need every agent prompt, tool trace, inter-branch message, token count and aggregation rule, and we plan to log all of these in follow-up runs so that the effects of decomposition, extra compute and information access can be separated.

*Written with Prayalankar and Prathosh AP. Comments and pointers to related work are welcome. Open an issue on [GitHub](https://github.com/Vandit-iisc).*
