---
date: 2026-06-18
title: "Learning from Preferences"
math: true
weight: 1
postType: review
linkTitle: "Learning from Preferences"
readingTime: 30
---

{{< katex />}}

# Learning from Preferences
{{< postbadges >}}

It can be easier to choose between two answers than to write a perfect answer or assign a meaningful numerical score. Preference learning starts from these comparisons: given the same prompt, which response would a person rather receive? This post connects a probabilistic model of that choice to reward learning and policy optimization.

Throughout, $x$ denotes a prompt and $y$ a complete response. We observe a dataset of comparisons,

{{< katex display=true >}}
\mathcal{D}_{\mathrm{pref}}
= \left\{\left(x^{(i)}, y_w^{(i)}, y_l^{(i)}\right)\right\}_{i=1}^{N},
\qquad y_w \succ y_l \mid x,
{{< /katex >}}

where $w$ and $l$ mean the preferred and dispreferred responses in an annotated pair. A preference label tells us their relative quality; it does not say that the winner is perfect or the loser is useless.

## Bradley-Terry Model

### From Scores to Comparisons

The [Bradley–Terry (BT) model](https://doi.org/10.1093/biomet/39.3-4.324) assigns a positive strength to each alternative and models a pairwise choice using their relative strengths (Bradley and Terry, 1952). For language responses, parameterize this strength as $u(x,y)=\exp(r(x,y))$, where $r$ is a scalar reward.

<div class="definition">
<strong>Definition 1.</strong> Under the Bradley–Terry model, the probability of preferring one response to another is

{{< katex display=true >}}
\begin{aligned}
p_r(y_a \succ y_b \mid x)
&= \frac{\exp(r(x,y_a))}{\exp(r(x,y_a))+\exp(r(x,y_b))} \\
&= \sigma\!\left(r(x,y_a)-r(x,y_b)\right),
\qquad \sigma(z)=\frac{1}{1+\exp(-z)}.
\end{aligned}
{{< /katex >}}
</div>

The reward difference is a log-odds: if $p=p_r(y_a\succ y_b\mid x)$, then $\log\frac{p}{1-p}=r(x,y_a)-r(x,y_b)$. Equal rewards give a preference probability of $1/2$; a difference of $\log 3$ gives $3/4$. Thus a score gap measures confidence in a comparison, rather than an absolute amount of human satisfaction.

### Learning a Reward Model

Let $r_\phi(x,y)$ be a neural reward model. Treating the observed comparisons as independent samples, maximum likelihood gives the negative log-likelihood loss,

{{< katex display=true >}}
\begin{aligned}
\Delta r_\phi(x,y_w,y_l)
&= r_\phi(x,y_w)-r_\phi(x,y_l), \\
\mathcal{L}_{\mathrm{RM}}(\phi)
&= -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}_{\mathrm{pref}}}
  \left[\log\sigma\!\left(\Delta r_\phi(x,y_w,y_l)\right)\right].
\end{aligned}
{{< /katex >}}

For a single comparison, $\partial\ell/\partial\Delta r_\phi=-\sigma(-\Delta r_\phi)$. Gradient descent therefore pushes the winner's score above the loser's, with a larger correction when the current ordering is wrong. The model learns from comparisons, but its output is a score for an individual prompt–response pair. This lets it score new responses without asking a human to compare every new pair.

### What Is Identifiable?

Adding the same prompt-dependent offset $c(x)$ to both rewards leaves every preference probability unchanged,

{{< katex display=true >}}
\bigl[r(x,y_a)+c(x)\bigr]-\bigl[r(x,y_b)+c(x)\bigr]
= r(x,y_a)-r(x,y_b).
{{< /katex >}}

Consequently, comparisons within a prompt cannot identify an absolute reward origin. Reward scale is different: multiplying rewards by a constant changes the probabilities when the sigmoid temperature is fixed. These two facts will matter when we introduce KL regularization and derive DPO.

BT also makes a substantive modeling assumption: one scalar ordering explains the choice probabilities for a fixed prompt. If $r(x,y_a)>r(x,y_b)>r(x,y_c)$, all three pairwise majority preferences follow that ordering. Individual annotations can still disagree or form cycles; the model does not make noisy observations transitive. Persistent population-level cycles, annotator-specific priorities, and explicit ties require more expressive models or additional modeling choices. A learned reward summarizes the comparisons collected under a particular annotation protocol.

## RL from Human Feedback

RLHF turns a learned preference score into a training signal for a policy. The standard language-model pipeline has three stages: supervised fine-tuning, reward modeling, and reinforcement learning, as in [Ouyang et al. (2022)](https://arxiv.org/abs/2203.02155). Here we focus on the basic objective; practical systems can add other training losses.

### 1. Supervised Fine-Tuning

Starting from a pretrained language model, fit human demonstrations with the usual conditional negative log-likelihood,

{{< katex display=true >}}
\mathcal{L}_{\mathrm{SFT}}(\theta)
= -\mathbb{E}_{(x,y)\sim\mathcal{D}_{\mathrm{SFT}}}
\left[\log\pi_\theta(y\mid x)\right].
{{< /katex >}}

This gives a policy $\pi_{\mathrm{SFT}}$ that can produce plausible task responses. Initialize the trainable policy from it, and keep a frozen copy as the reference policy $\pi_{\mathrm{ref}}$. The reference will give us a baseline distribution against which to measure changes.

### 2. Reward Modeling

Collect candidate responses and human comparisons, then fit $r_\phi$ with the BT loss above. In the basic pipeline, freeze this reward model during the subsequent RL stage. The policy generates text, while the reward model evaluates it; the reward model's parameters $\phi$ and the policy's parameters $\theta$ have different roles.

Human feedback enters through the comparison dataset. RL updates can then use the learned reward to evaluate newly sampled responses, without obtaining a fresh human judgment for each rollout.

### 3. Optimizing the Policy

Maximizing only the learned score can drive the policy toward responses where the reward model is unreliable. A common objective adds a KL penalty relative to the reference policy, following [Ziegler et al. (2019)](https://arxiv.org/abs/1909.08593):

{{< katex display=true >}}
\begin{aligned}
\max_\theta\ J(\theta)
&= \mathbb{E}_{x\sim\mathcal{D}_x}\Bigl[
  \mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)}[r_\phi(x,y)] \\
&\hspace{3.5em}
  -\beta D_{\mathrm{KL}}\!\left(
    \pi_\theta(\cdot\mid x)\,\|\,\pi_{\mathrm{ref}}(\cdot\mid x)
  \right)\Bigr], \qquad \beta>0,
\end{aligned}
{{< /katex >}}

where $\mathcal{D}_x$ is the training prompt distribution. The KL direction is policy-to-reference, with responses averaged under the current policy. For a fixed reward function, increasing $\beta$ makes deviations more expensive. This discourages excessive drift, although it cannot guarantee that the reward model remains accurate.

Expanding the KL term gives an equivalent expectation of a regularized sequence reward,

{{< katex display=true >}}
\begin{aligned}
\widetilde r_\theta(x,y)
&= r_\phi(x,y)
  -\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}, \\
J(\theta)
&= \mathbb{E}_{x\sim\mathcal{D}_x,\,y\sim\pi_\theta(\cdot\mid x)}
   [\widetilde r_\theta(x,y)].
\end{aligned}
{{< /katex >}}

For an autoregressive model, a complete response has log-probability

{{< katex display=true >}}
\log\pi_\theta(y\mid x)
= \sum_{t=1}^{T}\log\pi_\theta(y_t\mid x,y_{<t}).
{{< /katex >}}

We can therefore view generation as an episode: the state is the prompt plus the response prefix, the action is the next token, and the reward model supplies a terminal score. The log-ratio penalty decomposes into token-level terms. A sampled log-ratio can be negative; its expectation under the current policy is the nonnegative KL divergence.

[PPO](https://arxiv.org/abs/1707.06347) is one way to optimize this objective. It samples responses, estimates token advantages using a value function, and updates the policy with a clipped surrogate. Its importance ratio uses the policy that generated the rollout,

{{< katex display=true >}}
\rho_t(\theta)
= \frac{\pi_\theta(y_t\mid x,y_{<t})}
       {\pi_{\mathrm{old}}(y_t\mid x,y_{<t})}.
{{< /katex >}}

The rollout policy $\pi_{\mathrm{old}}$ is refreshed during training, whereas $\pi_{\mathrm{ref}}$ stays fixed in this setup. PPO clipping controls an individual update; the reference KL penalizes cumulative departure from the starting model. See [From Policy Gradient to PPO]({{< relref "/rl/from-policy-gradient-to-ppo.md" >}}) for the optimization details.

The practical difficulty is coordinating generation, reward evaluation, value estimation, and policy updates. Moreover, improving the learned score need not improve human judgments if the policy exploits errors in the reward model. This motivates asking whether comparisons can train the policy more directly.

## Direct Preference Optimization

[Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290), introduced by Rafailov et al. (2023), uses the relationship between a reward and its KL-regularized optimal policy to express preference likelihood directly in terms of a language model. Its standard offline form needs neither a separately trained reward model nor a rollout-and-PPO loop during preference training.

### From a Reward to Its Optimal Policy

Fix a prompt $x$ and a reward function $r$. Optimize the RLHF objective over response distributions $\pi(\cdot\mid x)$. Assume $\pi_{\mathrm{ref}}$ is positive on the response space considered and the following normalizing sum is finite:

{{< katex display=true >}}
Z_r(x)=\sum_y\pi_{\mathrm{ref}}(y\mid x)
\exp\!\left(\frac{r(x,y)}{\beta}\right).
{{< /katex >}}

The optimal distribution is an exponential tilt of the reference,

{{< katex display=true >}}
\boxed{\pi_r^*(y\mid x)
=\frac{\pi_{\mathrm{ref}}(y\mid x)}{Z_r(x)}
\exp\!\left(\frac{r(x,y)}{\beta}\right).}
{{< /katex >}}

{{% details "Deriving the optimal policy" %}}

For this fixed prompt, let $J_x(\pi)$ denote expected reward minus the KL penalty. Substitute the definition of $\pi_r^*$ and rearrange:

{{< katex display=true >}}
\begin{aligned}
J_x(\pi)
&=\sum_y\pi(y\mid x)
\left[r(x,y)-\beta\log\frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}\right] \\
&=\beta\log Z_r(x)
-\beta\sum_y\pi(y\mid x)\log\frac{\pi(y\mid x)}{\pi_r^*(y\mid x)} \\
&=\beta\log Z_r(x)
-\beta D_{\mathrm{KL}}\!\left(\pi(\cdot\mid x)\,\|\,\pi_r^*(\cdot\mid x)\right).
\end{aligned}
{{< /katex >}}

The first term is independent of $\pi$, and the KL divergence is minimized at zero. Thus $\pi=\pi_r^*$ maximizes the objective. This statement concerns unrestricted distributions; a finite neural policy class may only approximate the optimum.

{{% /details %}}

For a concrete example, suppose there are only two possible responses, with reference probabilities $1/2$ each, $\beta=1$, and rewards $\log 3$ and $0$. Their unnormalized policy weights become $3/2$ and $1/2$, so the optimal policy assigns probabilities $3/4$ and $1/4$. Raising a reward multiplies its reference probability before renormalization.

### Eliminating the Unknown Normalizer

Taking logarithms and solving for the reward gives

{{< katex display=true >}}
r(x,y)
=\beta\log\frac{\pi_r^*(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
+\beta\log Z_r(x).
{{< /katex >}}

Computing $Z_r(x)$ would require summing over the response space. But the BT likelihood uses two rewards for the same prompt. Their shared $\beta\log Z_r(x)$ cancels. We can therefore parameterize a reward representative directly by a trainable policy,

{{< katex display=true >}}
\widehat r_\theta(x,y)
=\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}.
{{< /katex >}}

This exploits the offset ambiguity from the BT section: preference observations do not require recovering the prompt-dependent normalization term. Define the policy's preference margin as

{{< katex display=true >}}
\begin{aligned}
m_\theta(x,y_w,y_l)
&=\widehat r_\theta(x,y_w)-\widehat r_\theta(x,y_l) \\
&=\beta\left[
\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
\right].
\end{aligned}
{{< /katex >}}

Substituting this margin into the BT loss yields

{{< katex display=true >}}
\boxed{\mathcal{L}_{\mathrm{DPO}}(\theta)
=-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}_{\mathrm{pref}}}
\left[\log\sigma\!\left(m_\theta(x,y_w,y_l)\right)\right].}
{{< /katex >}}

### What the Update Changes

For a single pair, differentiating $\ell=-\log\sigma(m_\theta)$ gives

{{< katex display=true >}}
\begin{aligned}
\nabla_\theta\ell
=-\beta\sigma(-m_\theta)\Bigl[
&\nabla_\theta\log\pi_\theta(y_w\mid x) \\
-&\nabla_\theta\log\pi_\theta(y_l\mid x)
\Bigr].
\end{aligned}
{{< /katex >}}

The frozen reference contributes to the margin but has no parameter gradient. The update favors a larger winner-to-loser probability ratio, with more weight on pairs whose implicit reward ordering is wrong. Shared model parameters mean that this does not guarantee the absolute probability of every winning response increases after an update.

Consider a reference assigning probabilities $0.1$ and $0.2$ to a winner and loser, respectively. The following policies give different margins when $\beta=1$; all remaining probability belongs to other responses.

| Policy | Winner probability | Loser probability | Margin | Pair loss |
| :--- | ---: | ---: | ---: | ---: |
| Reference initialization | 0.10 | 0.20 | 0 | 0.693 |
| More mass on the winner | 0.20 | 0.20 | $\log 2$ | 0.405 |
| Both decrease, loser decreases more | 0.05 | 0.05 | $\log 2$ | 0.405 |

The last two policies have the same loss on this pair. DPO constrains relative changes on observed comparisons, so a lower training loss alone does not establish better generation quality.

### Computing the Loss

With one sequence log-probability per response and model, the loss is small enough to implement directly:

```python
import torch.nn.functional as F

# Each input contains response log-probabilities, shape [batch_size].
policy_logratio = policy_chosen_logps - policy_rejected_logps
ref_logratio = (ref_chosen_logps - ref_rejected_logps).detach()
margin = beta * (policy_logratio - ref_logratio)
loss = -F.logsigmoid(margin).mean()
```

Compute these log-probabilities by summing next-token log-probabilities over the response, conditioning on the prompt and preceding response tokens. Mask out prompt and padding positions, handle the response-ending token consistently, and use the same tokenization for policy and reference. Averaging by response length would change the sequence-probability objective derived above. The reference stays frozen, and its log-probabilities can be cached when the dataset and preprocessing stay fixed.

At initialization, $\pi_\theta=\pi_{\mathrm{ref}}$ implies a zero margin for every pair and a loss of $\log 2$. This is a useful implementation check, even when the reference assigns very different probabilities to the two responses.

The derivation connects DPO to an ideal KL-regularized solution under the preference model. It does not guarantee identical outcomes to PPO with finite data and restricted neural models. Likewise, $\beta$ inherits its role from the KL objective, but the empirical DPO loss does not impose a hard KL bound. Data coverage, annotation quality, and generalization still determine whether the learned preferences transfer to new generations.

## RLHF v.s. DPO

## Iterative Preference Learning

## References

{{< references >}}
<li>Bradley, R. A., &amp; Terry, M. E. (1952). Rank analysis of incomplete block designs: I. The method of paired comparisons. <em>Biometrika</em>, 39(3–4), 324–345.</li>
<li>Ziegler, D. M., Stiennon, N., Wu, J., Brown, T. B., Radford, A., Amodei, D., Christiano, P., &amp; Irving, G. (2019). Fine-tuning language models from human preferences. arXiv:1909.08593.</li>
<li>Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., et al. (2022). Training language models to follow instructions with human feedback. <em>Advances in Neural Information Processing Systems</em>, 35, 27730–27744.</li>
<li>Schulman, J., Wolski, F., Dhariwal, P., Radford, A., &amp; Klimov, O. (2017). Proximal policy optimization algorithms. arXiv:1707.06347.</li>
<li>Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., &amp; Finn, C. (2023). Direct preference optimization: Your language model is secretly a reward model. <em>Advances in Neural Information Processing Systems</em>, 36, 53728–53741.</li>
<li>Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., ... & Anadkat, S. (2024). Gpt-4 technical report. arXiv 2023. arXiv preprint arXiv:2303.08774.</li>

{{< /references >}}
