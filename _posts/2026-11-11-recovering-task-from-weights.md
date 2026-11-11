---
layout: post
title: "Can an LLM Read a Neural Network? Recovering a Classifier's Task from Its Weights"
description: "We gave LLMs the raw weights of a small digit classifier and asked what it was trained to do. The wrong answers, a two-bit hint, and the maths of why weights are hard to read."
authors:
  - name: Vandit Patel
    affiliation: (Indian Institute of Science)
  - name: Prayalankar
    affiliation: (Stryker AI Research)
  - name: Prathosh AP
    affiliation: (Indian Institute of Science)
---

Suppose someone hands you 100,609 floating-point numbers. They tell you these are the weights of a small network that takes a 28×28 grayscale image and outputs one bit. There are no training data, no labels and no description of the task. Could you work out what the network was trained to do?

We put that question to large language models, in two ways. A single long call to **GPT-6 Astra** worked out that the network looks at handwritten digits, but it guessed the wrong rule: even vs. odd. A team of **Claude** agents, working without hints, settled confidently on a different wrong rule, "zero vs. everything else". Then they were told one small fact, that each class contains five digits. After that, their candidates came within one or two digits of the truth, and one of the four agents found the exact rule.

This post tells that story in order, including the wrong turns, which turned out to be the most informative part. Then it works through the maths of *why* a trained network's weights are so hard to read, and checks every claim the agents made against the network itself.

## Contents

1. [The network](#the-network)
2. [The ground rules, and what each system saw](#the-ground-rules-and-what-each-system-saw)
3. [Act 1: one long call guesses parity](#act-1-one-long-call-guesses-parity)
4. [Act 2: blind agents see a zero](#act-2-blind-agents-see-a-zero)
5. [Act 3: a two-bit hint and four angles](#act-3-a-two-bit-hint-and-four-angles)
6. [reading weights](#reading-weights)
7. [Checking the agents against the network](#checking-the-agents-against-the-network)
8. [What we learned](#what-we-learned)

## The network

The idea comes from the *Weight-Space Programming* experiments by Takehi, Teoh, Zhang and Khattab (MIT CSAIL). There, LLMs inspect, describe and edit neural-network parameters directly. In one experiment, an LLM reads the weights of a game-playing policy and names the game. We ask the same question of a much plainer object:

<figure>
  <img src="{{ '/assets/img/f01_pipeline.png' | relative_url }}" alt="Pipeline: 28x28 image, flattened to 784 inputs, 128 ReLU hidden units, one raw logit, thresholded at zero">
  <figcaption>Figure 1. The classifier: one hidden layer of 128 ReLU units and a single output logit.</figcaption>
</figure>

With $$x = \mathrm{vec}(X/255) \in \mathbb{R}^{784}$$, the network computes

$$
h(x) = \mathrm{ReLU}(W_1 x + b_1), \qquad z(x) = W_2\, h(x) + b_2, \qquad \hat y(x) = \mathbb{1}\{z(x) > 0\}.
$$

It was trained with `BCEWithLogitsLoss`, so $$z$$ is a raw logit and $$\sigma(z) > 0.5$$ means class 1. Flattened pixel $$i$$ sits at row $$\lfloor i/28 \rfloor$$ and column $$i \bmod 28$$. The whole post reads the network one hidden unit at a time, so here is the notation once:

| Symbol | Meaning |
|---|---|
| $$w_j = W_1[j,:]^\top \in \mathbb{R}^{784}$$ | unit $$j$$'s input weights; reshaped to 28×28, a "template" |
| $$b_j = b_1[j]$$ | unit $$j$$'s bias |
| $$v_j = W_2[0,j]$$ | unit $$j$$'s output weight, its "vote": $$v_j > 0$$ pushes toward class 1 |
| $$a_j(x) = w_j^\top x + b_j$$ | pre-activation |
| $$s_j(x) = \mathbb{1}[a_j(x) > 0]$$ | gate: is the unit on? |
| $$c_j(x) = v_j \max(0, a_j(x))$$ | the unit's contribution to the logit, so $$z = \sum_j c_j + b_2$$ |
| $$T = \sum_j v_j w_j$$ | the vote-weighted sum of all templates, used heavily by the agents |

The task the network was actually trained on is a map $$g : \{0,\dots,9\} \to \{0,1\}$$:

$$
g(d) = 1 \iff d \in \{0, 2, 3, 5, 7\},
$$

that is, **primes plus zero** against everything else. We score a guess by how many of the ten digits it places on the correct side, out of 10. Exact recovery means all ten. Two baselines help calibrate the scores. A random five-against-five split gets 5/10 right on average, because each digit is a coin flip. And it hits the exact set with probability $$1/\binom{10}{5} = 1/252 \approx 0.4\%$$.

## The ground rules, and what each system saw

The agents were held to strict rules:

- No forward passes: they could not run the network on any input, real or synthetic.
- They could not look at the example image.
- They could not compute dot products between the weights and imagined digits, since that is a forward pass in disguise.

What they could do was analyse and plot the weights, and use their general knowledge of what digits look like to *interpret* what they saw. The two systems were not given the same information, which matters for any comparison:

| | Single call | Agent system |
|---|---|---|
| Model | GPT-6 Astra (high reasoning effort) | Claude Opus 5.5, in Claude Code |
| Structure | one request, no tools | 4 main agents × 3 subagents = 12, after earlier single-agent rounds |
| Tools | none | Python (numpy, matplotlib) on the weights only |
| Weights | full precision | full precision |
| Labelled example | yes, one image (a 0, class 1) | no, forbidden |
| "Each class has 5 digits" | no | yes, in the final round |
| "The data is plain MNIST" | no | yes, in the final round, after agents inferred it |
| Compute | 669,642 input and 14,546 output tokens | not matched; much larger |

So this is not a controlled comparison. The model, the information and the compute all differ. We report what each configuration did, not which approach is better.

## Act 1: one long call guesses parity

Astra received everything in one prompt: every weight, the conventions above, and one labelled image. The first attempt ran out of its 4,096-token output budget mid-reasoning and returned nothing. The second, with 32,768 tokens, completed. Astra correctly described the example as a handwritten zero and answered

$$
\text{class 1} = \{0,2,4,6,8\}, \qquad \text{class 0} = \{1,3,5,7,9\}
$$

at 42% confidence, calling parity tentative. This was not a blind guess. It cited real weights, for example $$W_1[2,349] \approx -0.635$$ with $$v_2 \approx +0.478$$, and these values match the file. But parity places only 0, 1, 2 and 9 correctly. That's **4/10, below the 5/10 a random split gets on average.** Astra read the weights and then reached for the most familiar binary rule over digits.

## Act 2: blind agents see a zero

The agents started with no hints at all.

**Round 1** was a single agent working blind. It found that the strongest class-1 units share a distinctive shape: a narrow negative stroke flanked by positive weights. Its cross-section averaged about $$+0.03 / -0.25 / +0.03$$, which means "ink beside this line, but not on it". The class-0 units had positive biases and strongly negative weights, so they are on by default and switched off by ink. The agent then summed each unit's strongest lobes, weighted by its vote:

<figure>
  <img src="{{ '/assets/img/f02_lobes.png' | relative_url }}" alt="Four heatmaps: class-1 units forbid ink in the centre and want it on a ring; class-0 units are switched off by ink on the same ring">
  <figcaption>Figure 2. The agents' "lobe maps", reproduced from their script. Class-1 units want ink on a large ring (second panel) and forbid it in the middle (first). Ink on the same ring switches class-0 units off (third). A ring with an empty centre looks like a zero.</figcaption>
</figure>

A ring of ink around an empty centre pushes toward class 1. The radial profile of $$T$$ agrees: about $$-1.3$$ near the centre and $$+0.94$$ at radius 12–14. The agent concluded "class 1 is the digit 0, on rotated data", at 55–65%.

**Round 2** added an independent second opinion. A separate subagent, deliberately not told the first answer, made two findings:

- **Exactly 67 pixels have never been trained**, matching the 67 pixels that are blank in every MNIST training image.
- The mask of untrained pixels has no mirror or transpose symmetry, so the data was **not** rotated or flipped. This overturned the rotation claim.

It too concluded "0 vs. all other digits", at about 65%. The network really does reward a ring with an empty centre, because 0 really is in class 1. It just isn't the whole rule. 

## Act 3: a two-bit hint and four angles

We then told the agents one fact: **each class contains exactly five digits.** It is worth measuring how much information that is. Before the hint, any labelled split of ten digits into two nonempty classes was possible. After it, only the five-against-five ones remain:

$$
\log_2 \frac{2^{10} - 2}{\binom{10}{5}} = \log_2 \frac{1022}{252} \approx 2 \text{ bits}.
$$

Two bits immediately ruled out "0 vs. rest". A single agent then reread the same features as a five-against-five split and proposed parity, the same wrong answer as Astra, at about 55%. It flagged its own weakest link: one strong class-1 unit is switched off by a bar near row 16, roughly where a 4's crossbar sits.

The final round was planned rather than improvised. Four main agents, each with three subagents, attacked the problem from different angles, all from the same brief:

| Agent | Angle |
|---|---|
| A | **Training dynamics.** Under BCE, weight updates are sums of training images, so $$T$$ should resemble the mean class-1 image minus the mean class-0 image. |
| B | **Unit by unit.** Read the ~40 strongest units as detectors for look-alike pairs on opposite sides, then score all 126 five-against-five splits. |
| C | **Hypothesis critic.** List natural splits (0–4 vs. 5–9, primes, loops, curved vs. angular, …), predict what the weights should show for each, and try to refute them. |
| D | **Digit by digit.** Match each digit's distinctive strokes to units and tally their votes. |

Each agent reported a probability that each digit is in class 1. Here they are against the truth:

<figure>
  <img src="{{ '/assets/img/f03_agent_probs.png' | relative_url }}" alt="Grid of each agent's probability that each digit is in class 1, coloured from orange to blue, with wrong-side cells outlined and a Brier score per agent">
  <figcaption>Figure 3. Each agent's P(digit in class 1). Dashed cells are on the wrong side of 0.5. The Brier score \(\frac{1}{10}\sum_d (p_d - g(d))^2\) is 0 for perfect answers and 0.25 for coin flips.</figcaption>
</figure>

All four agree on the easy part: 2 and 3 go to class 1, and 1 and 4 to class 0. Everything hard is concentrated in 5, 6, 7 and 8. The agents' top picks, together with every earlier answer, line up like this:

<figure>
  <img src="{{ '/assets/img/f04_scoreboard.png' | relative_url }}" alt="Scoreboard of every answer: Astra and the single hinted agent 4/10, blind agents 6/10, agents A and D 8/10, B 6/10, C 10/10, the averaged final answer 6/10">
  <figcaption>Figure 4. Every answer in the experiment, digit by digit. Agent C's top pick is exact. The system's final answer, from averaging the four, is B's split, scoring 6/10.</figcaption>
</figure>

There is a lesson in how the final answer was chosen. Agent C was by far the best calibrated (Brier 0.111) and its top split is exactly right, although even C was close to a coin flip on 5 and 8. Agent B was *worse than a coin flip* (0.286), yet averaging the four probabilities produced B's split, {0, 2, 3, 6, 8}, at only 25–30% confidence. Averaging pulls the ensemble toward agents that are confidently wrong, and without the ground truth there is no way to know which agent to trust. The hint took the system from a wrong answer to candidates within one or two digits of the truth. It still did not let the system *choose* the right one.

## Reading weights

Why is it so hard to read this rule off the weights? 

### 1. Weights are made of training images

Write the per-example loss as $$\ell = -\big[y \log \sigma(z) + (1-y)\log(1-\sigma(z))\big]$$. Its derivative with respect to the logit simplifies to the prediction error, $$\partial \ell / \partial z = \sigma(z) - y =: e$$, because the sigmoid's derivative $$\sigma(1-\sigma)$$ cancels the denominators of the log-loss. From the definitions above,

$$
\frac{\partial z}{\partial w_j} = v_j\, s_j(x)\, x ,
\qquad\text{so}\qquad
g_j = \sum_{n \in \text{batch}} e_n\, v_j\, s_j(x_n)\, x_n
$$

is the gradient for $$w_j$$ on one batch: an error-weighted sum of *input images*. With plain SGD, each step subtracts $$\eta\, g_j$$, and the trained template would be its initialisation plus a weighted sum of training images. This network was trained with Adam, which changes the step in two ways. It replaces $$g_j$$ by a running average $$m_j$$ of recent gradients, and it rescales every pixel's step by its own positive factor $$d_{j,i} = 1/(\sqrt{\hat v_{j,i}} + \epsilon)$$:

$$
w_j^{(\text{end})} = w_j^{(0)} - \eta \sum_{\text{steps } t} d_j^{(t)} \odot m_j^{(t)} .
$$

**So every trained template is its initialisation plus a sum of training images, each rescaled pixel by pixel.** The images still decide the sign and shape of every update; Adam only changes how large each pixel's step is. Three of the agents' findings follow directly.

**Frozen pixels.** If pixel $$i$$ is blank in every training image, then $$x_{n,i} = 0$$ for all $$n$$. The gradient for $$w_{j,i}$$ is then zero at every step, so Adam's running average for that pixel is zero and its step is exactly zero (there is no weight decay). The weight stays at its initial value. PyTorch initialises this layer from $$\mathcal{U}(-1/\sqrt{784}, 1/\sqrt{784}) = \mathcal{U}(-1/28, 1/28)$$. A pixel whose 128 weights all still lie within $$\pm 1/28$$ is therefore almost certainly a pixel that was never inked. The agents counted exactly 67 such pixels, the known MNIST number, and all 67 are blank in every one of our sample images. A dataset leaves a fingerprint in the weights.

**Dead units.** If unit $$j$$ is off for every training image at every step of training, then $$s_j(x_n) = 0$$ throughout, its gradient is always zero, and the whole row never moves. Its output weight doesn't move either, because $$\partial z/\partial v_j = \max(0, a_j) = 0$$. Four units, 11, 38, 106 and 114, sit exactly at the norm expected at initialisation, $$\sqrt{784 \cdot (1/28)^2/3} \approx 0.58$$. Their output weights are still inside the initial range $$\pm 1/\sqrt{128} \approx 0.088$$, and none of them fires on any of our 50 images. Every other unit grew well beyond its initial norm: by a median of about 5.6×, ranging from 2.3× to 9.1×.

<figure>
  <img src="{{ '/assets/img/f05_frozen.png' | relative_url }}" alt="Left: the 67 frozen pixels at the corners and edges of the image. Right: weight norms of all 128 units; four sit at the initial value 0.58, the rest are around 3">
  <figcaption>Figure 5. Two fingerprints of the training data. Left: the 67 pixels no image ever inked. Right: four units that never switched on and so never trained.</figcaption>
</figure>

**Hard examples dominate.** Agent A turned this into a prediction. As a heuristic, ignore Adam's averaging, treat the $$v_j$$ as slowly varying, and sum over units:

$$
T = \sum_j v_j w_j \;\approx\; \sum_j v_j w_j^{(0)} \;-\; \eta \sum_{\text{steps } t}\sum_n e_n \sum_j v_j^2\, s_j(x_n)\, \big(d_j^{(t)} \odot x_n\big) .
$$

Within every step, each image enters with weight proportional to its error $$e_n$$. Adam keeps the overall step size roughly constant even as the errors shrink, so late in training the steps are as large as ever but are driven almost entirely by the few images that are still misjudged. Early in training every image has a large error, and $$T$$ resembles the mean class-1 image minus the mean class-0 image. But once an image is classified confidently, $$e_n \to 0$$ and it stops contributing. The final weights are therefore shaped by the *hard* images near the boundary, much like support vectors in an SVM. Agent A's approximation "$$T \approx$$ mean class-1 image − mean class-0 image" holds only weakly: on our samples the correlation is 0.12. **A well-trained network stores the differences between look-alike digits, not what each class looks like.** That is why "what does class 1 look like?" has no direct answer in the weights.

### 2. Units are detectors with vetoes

The "ink beside the line, but not on it" shape fits this picture. Split unit $$j$$'s pre-activation by region:

$$
a_j(x) = \underbrace{\sum_{i \in \text{flanks}} w_{j,i}\, x_i}_{\text{small positive weights, many pixels}} \;+\; \underbrace{\sum_{i \in \text{slit}} w_{j,i}\, x_i}_{\text{large negative weights, few pixels}} \;+\; b_j .
$$

Ink spread along the flanks slowly accumulates positive evidence, while a little ink in the slit vetoes the unit outright. On real data these units turn out to be sharp digit specialists, and the slit sits exactly where a look-alike digit would put ink:

<figure>
  <img src="{{ '/assets/img/f06_slits.png' | relative_url }}" alt="Ten weight templates of top class-1 units, each with the outline of the average digit it fires on: three 3-detectors, four 2-detectors, three 7-detectors">
  <figcaption>Figure 6. The strongest class-1 units, each labelled with the digit it actually fires on and overlaid with that digit's average shape. The dark slits sit in the gaps the digit leaves open.</figcaption>
</figure>

Units 28, 61 and 69 are 3-detectors whose slits sit in the open left side of the 3, roughly where an 8 would close the loop. Unit 61 fires on none of our five 8s, and units 28 and 69 each fire on only one. Units 63, 86, 100 and 46 detect 2s, and units 13, 30 and 102 detect 7s. Each unit encodes a *contrast* between a class-1 digit and its class-0 look-alike, as the training dynamics predicts.

### 3. The network is piecewise linear, and the gates decide

ReLU is nonlinear, but only at one point: it is the line of slope 0 for $$t \le 0$$ and the line of slope 1 for $$t > 0$$. Collect the gates into $$S(x) = \mathrm{diag}(s_1(x), \dots, s_{128}(x))$$. Then

$$
z(x) = W_2\, S(x)\,(W_1 x + b_1) + b_2 .
$$

The 128 hyperplanes $$a_j(x) = 0$$ cut the input space into convex cells. Inside any one cell no gate changes, $$S$$ is constant, and $$z$$ is **exactly affine**. All the nonlinearity lives on the cell walls. You can see this by walking in a straight line from one digit to another:

<figure>
  <img src="{{ '/assets/img/f08_kinks.png' | relative_url }}" alt="Logit along a straight path from a 7 to a 1: a piecewise-linear curve with 57 kinks, crossing zero around t = 0.6; below, the number of active units drops from about 63 to about 44">
  <figcaption>Figure 7. The logit along the straight line from a 7 to a 1. Between kinks the curve is exactly straight. At each of the 57 kinks a unit switches on or off. The prediction flips from class 1 to class 0 around \(t \approx 0.6\).</figcaption>
</figure>

This matters for reading weights. Within the cell containing $$x$$, the gradient is

$$
\nabla_x z = \sum_j v_j\, s_j(x)\, w_j ,
$$

a vote-weighted sum of only the *active* units' templates. The agents' map $$T = \sum_j v_j w_j$$ is the same sum with **every** gate set to 1. A real image never switches on all the units:

<figure>
  <img src="{{ '/assets/img/f07_gating.png' | relative_url }}" alt="Left: histogram of active units per image, between 24 and 65, mean 47. Right: histogram of per-image correlation between T and the true gradient, mean 0.79; the correlation with the average gradient is 0.97">
  <figcaption>Figure 8. Left: each image activates about 47 of 128 units. Right: the agents' map \(T\) matches the <em>average</em> gradient almost perfectly (0.97), but any single image's gradient less well (0.79 on average).</figcaption>
</figure>

$$T$$ is an excellent summary of where evidence lives *on average*. As a classifier it is useless: the "all units on" logit, $$x^\top T + \sum_j v_j b_j + b_2$$, predicts class 0 for all 50 images. **The decision is made by which units switch on, and that depends on the input.** Weights alone can't show which gates a given digit opens. This is the deepest reason the agents could recover the network's general structure but not its exact rule.

## Checking the agents against the network

Everything so far used only the weights. Now we break the agents' rules on purpose. We ran the network on the 50 sample images and checked their claims against its real behaviour. The sample file has only binary labels, so we read each digit by eye. Every reading agrees with its label under the primes-plus-zero rule, except one scribble we could not identify and left out. With only a few images per digit, treat per-digit numbers as small-sample.

**The network does what we said, and its hard digits are the agents' hard digits.** It classifies all 50 correctly. The closest calls are a 6 and an 8 just below zero, and the smallest class-1 margins belong to 5 and 7:

<figure>
  <img src="{{ '/assets/img/f09_margins.png' | relative_url }}" alt="Logits of all 50 samples grouped by digit: class-1 digits above zero, class-0 below; closest calls are one 6 and one 8">
  <figcaption>Figure 9. The logit of every sample, grouped by digit. The digits nearest the boundary (5, 6, 7, 8) are the ones the agents disagreed on.</figcaption>
</figure>

**Class 0 is the default.** For a blank image, $$z(\mathbf{0}) = \sum_j v_j \max(0, b_j) + b_2 \approx -0.99$$. Of the 128 units, 56 have $$v_j < 0$$ and $$b_j > 0$$: they are on for an empty canvas and push toward class 0 until ink switches them off. The agents' account of class-0 units as "on by default" is confirmed.

**One decision, unit by unit.** Because $$z = \sum_j c_j(x) + b_2$$ exactly, we can watch a single decision being assembled:

<figure>
  <img src="{{ '/assets/img/f11_waterfall.png' | relative_url }}" alt="Waterfall of the logit for one handwritten 3: output bias, then the largest positive and negative unit contributions, summing to z = 9.37">
  <figcaption>Figure 10. How the network reaches \(z = 9.37\) for one 3. Only 47 units are active, and a handful of 3-specialists (units 17, 96, 69) do most of the work against several class-0 units.</figcaption>
</figure>

**Which pixels do the work.** Inside a cell the network is affine, so the logit splits exactly, with no approximation, into per-pixel terms plus a bias term:

$$
z(x) = \sum_i x_i\, [\nabla_x z]_i \;+\; \sum_j v_j\, s_j(x)\, b_j + b_2 .
$$

(With a smooth activation like tanh, "gradient × input" would only be a local linear approximation.) Averaging the pixel term over each digit:

<figure>
  <img src="{{ '/assets/img/f10_attribution.png' | relative_url }}" alt="Ten heatmaps, one per digit, showing pixels pushing toward class 1 in blue and class 0 in orange">
  <figcaption>Figure 11. Exact pixel attribution, averaged per digit. The 7's top bar and the 3's upper arc push hard toward class 1. A 1's central stroke and a 9's loop-to-stem junction push toward class 0.</figcaption>
</figure>

**The agents' regions, erased.** The agents summarised the weights in five regions using a voting rule. For each region, they summed $$\lvert v_j \rvert$$ over units whose mean weight there is above $$+0.02$$ ("want ink") or below $$-0.02$$ ("forbid ink"). We then tested each region directly. For every image, we set the region's pixels to zero, $$\tilde X = M_R \odot X$$, re-ran the network, and recorded $$\Delta z = f(\tilde X) - f(X)$$.

| Region (rows, cols, inclusive) | Class-1 units want / forbid | Class-0 units want / forbid | Images with ink there | Mean $$\Delta z$$ when erased | Predictions flipped |
|---|---|---|---|---|---|
| R1 far bottom-left (19–24, 1–5) | 32.3 / 0.8 | 0.0 / 15.4 | 6 / 50 | −0.7 | 0 |
| R2 centre column (8–20, 13–15) | 3.9 / 18.1 | 6.2 / 2.9 | 50 / 50 | +3.7 (+13.5 for 1s) | 6 |
| R3 very top-right (1–3, 15–22) | 3.1 / 26.6 | 9.5 / 1.8 | 6 / 50 | +0.5 | 0 |
| R4 left-mid band (10–13, 2–15) | 5.0 / 22.7 | 0.7 / 11.0 | 48 / 50 | +6.6 | 10 |
| R5 lower-left loop (16–20, 7–11) | 8.1 / 19.5 | 6.3 / 6.1 | 41 / 50 | +1.5 | 2 |

The mean change is averaged over the images that have ink in the region, since erasing a blank region changes nothing. A flip is any image whose predicted class changes. A positive $$\Delta z$$ means the erased ink had been pushing toward class 0. One caveat: a digit with a block cut out of it is not natural handwriting, so this measures the network's response to that edit rather than an absolute "importance".

<figure>
  <img src="{{ '/assets/img/f12_regions.png' | relative_url }}" alt="Left: the five regions drawn over the mean digit. Middle: net weight vote per region. Right: the effect of erasing each region; the far bottom-left and top-right barely matter, the left-mid band matters most">
  <figcaption>Figure 12. What the weights say about each region (middle) against what erasing it actually does (right, plotted as \(-\Delta z\) so both panels point toward the class the ink supports).</figcaption>
</figure>

The comparison is instructive. The weights point the right way in every region, but their *strength* is a poor guide:

- **R1, the far bottom-left**, has the agents' largest weight vote, by far. Yet only 6 of 50 real digits put ink there, so it barely affects decisions. **R3, the top-right**, is the same story.
- **R4, the left-mid band**, has one of the *weakest* votes but is the region that matters most. Erasing it raises the logit by 6.6 on average and flips 10 predictions. This is where a 4's crossbar and the bottom of a 9's loop sit.

Large weights in places the data never reaches are invisible to the network's actual decisions. That is the trap of reading weights without inputs.

**The units the agents named.** Our first write-up named specific units as evidence. Testing them:

<figure>
  <img src="{{ '/assets/img/f13_named_units.png' | relative_url }}" alt="Heatmap of mean contribution per digit for nine named units: units 69 and 17 fire strongly on 3s; units 23 and 96 support 7s; unit 27 never fires on 7s">
  <figcaption>Figure 13. Mean contribution \(c_j(x)\) per digit for the units named as evidence. All nine have \(v_j > 0\), so each can only push toward class 1.</figcaption>
</figure>

The 3-vs-8 claim holds up: units 69 and 17 contribute more than 2 to the logit on a 3 and about 0.1–0.2 on an 8. The 7 claims do not. Unit 27 never fires on a 7, and units 23 and 96, said to oppose the 7's top bar, are among the 7's strongest supporters. A unit can have negative weights on part of a stroke and still fire for the digit as a whole.

## What we learned

**1. The overall structure is readable but the underlying rule that governs the neural network is not** From weights alone, the agents recovered a lot:

- the dataset, from the 67 frozen pixels;
- that the data was not augmented, from the mask's asymmetry;
- the four dead units;
- the default class;
- the families of detectors.

The maths says why the exact rule stays hidden. Trained weights encode error-weighted *differences* between look-alike digits, and the decision depends on which gates an input opens, which no weight shows on its own.

**2. A coherent story is not a correct one.** "A ring with an empty centre, so it's 0" fit every feature the blind agents looked at, and two independent analyses converged on it. Parity convinced both Astra and a hinted agent. Familiar answers are attractive even when evidence points elsewhere.

**3. Hints are data and that changes the prediction** Two bits of information, "five digits per class", changed the outcome more than any amount of analysis. Any claim about what an LLM can read from weights has to state exactly what it was told.

**4. We cant always average out all the agents answer** The best-calibrated agent was right and the worst was worse than chance. Averaging chose the worst agent's answer.

**5. Check weight-reading against inputs.** Every weight-based claim we tested had a simple behavioural test, and several failed it. Explanations from weights are hypotheses until activations on real data confirm them.

**What a fair comparison would need.** Same model for single-call and agentic runs, same information (hint or no hint, example or no example), matched compute budgets, and several random seeds for both the network and the agents. We'd also want to know whether calibration-weighted aggregation, or a critic that tests each agent's specific claims, would have selected agent C.

## Reproducibility

Everything in this post can be regenerated from the companion repository, **[Vandit-iisc/Recovering-Neural-Network-Task](https://github.com/Vandit-iisc/Recovering-Neural-Network-Task)**:

- `data/` holds the classifier's weights, the 50 sample images, and our by-eye digit readings.
- `astra/` holds the exact prompt sent to GPT-6 Astra and its settings.
- `agents/` holds the agents' weights-only analysis script and the shared brief for the four-agent round, which states the hint explicitly.
- `analysis/make_figures.py` builds every figure in this post, and `analysis/region_erasure.py` prints the region table.

A fuller record would include every agent and subagent transcript, tool trace and token count, so that the effects of decomposition, extra compute and extra information could be separated.

Comments and pointers to related work are welcome. 
