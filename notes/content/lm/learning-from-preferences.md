---
date: 2026-06-18
title: "Learning from Preferences"
math: true
weight: 1
postType: review
linkTitle: "Learning from Preferences"
readingTime: 20
---

{{< katex />}}

# Learning from Preferences
{{< postbadges >}}

Preference learning converts pairwise judgments into reward estimates and policy updates. We examine the Bradley–Terry model, KL-regularized RLHF, DPO, and the effects of model parameterization and iterative feedback.

Let $x$ denote a prompt and $y$ a complete response. The preference dataset is

{{< katex display=true >}}
\mathcal{D}_{\mathrm{pref}}
= \left\{\left(x^{(i)}, y_w^{(i)}, y_l^{(i)}\right)\right\}_{i=1}^{N},
\qquad y_w \succ y_l \mid x,
{{< /katex >}}

where $y_w$ and $y_l$ are the preferred and dispreferred responses, respectively.

## Bradley-Terry Model

### From Scores to Comparisons

The [Bradley–Terry (BT) model](https://doi.org/10.1093/biomet/39.3-4.324) parameterizes response strength as $u(x,y)=\exp(r(x,y))$ for a scalar reward $r$.

<div class="definition">
<strong>Definition 1 (Bradley–Terry).</strong> Pairwise preference probabilities satisfy

{{< katex display=true >}}
\begin{aligned}
p_r(y_a \succ y_b \mid x)
&= \frac{\exp(r(x,y_a))}{\exp(r(x,y_a))+\exp(r(x,y_b))} \\
&= \sigma\!\left(r(x,y_a)-r(x,y_b)\right),
\qquad \sigma(z)=\frac{1}{1+\exp(-z)}.
\end{aligned}
{{< /katex >}}
</div>

For $p=p_r(y_a\succ y_b\mid x)$, reward differences encode log-odds: $\log\frac{p}{1-p}=r(x,y_a)-r(x,y_b)$. Equal rewards imply $p=1/2$; a gap of $\log 3$ implies $p=3/4$.

### Learning a Reward Model

For independent comparisons, maximum-likelihood estimation of $r_\phi$ minimizes

{{< katex display=true >}}
\begin{aligned}
\Delta r_\phi(x,y_w,y_l)
&= r_\phi(x,y_w)-r_\phi(x,y_l), \\
\mathcal{L}_{\mathrm{RM}}(\phi)
&= -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}_{\mathrm{pref}}}
  \left[\log\sigma\!\left(\Delta r_\phi(x,y_w,y_l)\right)\right].
\end{aligned}
{{< /katex >}}

The derivative $\partial\ell/\partial\Delta r_\phi=-\sigma(-\Delta r_\phi)$ favors larger preferred-response margins, with greater weight on incorrectly ordered pairs. The resulting model scores individual prompt–response pairs.

### What Is Identifiable?

BT is invariant to prompt-dependent reward offsets:

{{< katex display=true >}}
\bigl[r(x,y_a)+c(x)\bigr]-\bigl[r(x,y_b)+c(x)\bigr]
= r(x,y_a)-r(x,y_b).
{{< /katex >}}

Comparisons cannot identify prompt-dependent reward offsets. Reward scale remains consequential at a fixed sigmoid temperature.

Scalar rewards impose a transitive population ordering, although individual annotations may contain cycles. Systematic preference cycles, annotator heterogeneity, and explicit ties require extensions to the basic model.

## RL from Human Feedback

The standard RLHF pipeline comprises supervised fine-tuning, reward estimation, and policy optimization ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)).

### 1. Supervised Fine-Tuning

Fine-tune a pretrained model on demonstrations by minimizing

{{< katex display=true >}}
\mathcal{L}_{\mathrm{SFT}}(\theta)
= -\mathbb{E}_{(x,y)\sim\mathcal{D}_{\mathrm{SFT}}}
\left[\log\pi_\theta(y\mid x)\right].
{{< /katex >}}

Initialize the trainable policy from the fitted model $\pi_{\mathrm{SFT}}$ and retain a frozen reference $\pi_{\mathrm{ref}}=\pi_{\mathrm{SFT}}$.

### 2. Reward Modeling

Fit $r_\phi$ to human comparisons using the BT loss, then freeze it during policy optimization. This learned evaluator scores new rollouts without requiring a human judgment for each response.

### 3. Optimizing the Policy

Maximize expected reward with a KL penalty that discourages departure from the reference ([Ziegler et al., 2019](https://arxiv.org/abs/1909.08593)):

{{< katex display=true >}}
\begin{aligned}
J(\theta)
&= \mathbb{E}_{x\sim\mathcal{D}_x}\Bigl[
  \mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)}[r_\phi(x,y)] \\
&\hspace{3.5em}
  -\beta D_{\mathrm{KL}}\!\left(
    \pi_\theta(\cdot\mid x)\,\|\,\pi_{\mathrm{ref}}(\cdot\mid x)
  \right)\Bigr], \qquad \beta>0,
\end{aligned}
{{< /katex >}}

Here {{< katex >}}\mathcal{D}_x{{< /katex >}} is the prompt distribution. The policy-to-reference KL is averaged under $\pi_\theta$; larger $\beta$ strengthens regularization for a fixed reward.

Equivalently, define the regularized sequence reward

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

Autoregressive factorization gives

{{< katex display=true >}}
\log\pi_\theta(y\mid x)
= \sum_{t=1}^{T}\log\pi_\theta(y_t\mid x,y_{<t}).
{{< /katex >}}

Generation is thus an episode with prefix states, token actions, a terminal reward, and token-level log-ratio penalties. Individual penalties may be negative; their expected sum is the nonnegative KL term.

[PPO](https://arxiv.org/abs/1707.06347) optimizes a clipped surrogate using estimated token advantages and the rollout-policy ratio

{{< katex display=true >}}
\rho_t(\theta)
= \frac{\pi_\theta(y_t\mid x,y_{<t})}
       {\pi_{\mathrm{old}}(y_t\mid x,y_{<t})}.
{{< /katex >}}

The refreshed rollout policy $\pi_{\mathrm{old}}$ differs from the fixed reference $\pi_{\mathrm{ref}}$: clipping moderates local updates, while the reference KL penalizes cumulative drift. See [From Policy Gradient to PPO]({{< relref "/rl/from-policy-gradient-to-ppo.md" >}}). This pipeline incurs rollout and value-estimation costs and remains vulnerable to reward-model errors.

## Direct Preference Optimization

[DPO](https://arxiv.org/abs/2305.18290) reparameterizes rewards through their KL-regularized optimal policies, enabling direct preference fitting without a separate reward model or online rollouts during offline training (Rafailov et al., 2023).

### From a Reward to Its Optimal Policy

For a fixed prompt and reward, assume full reference support and a finite partition function

{{< katex display=true >}}
Z_r(x)=\sum_y\pi_{\mathrm{ref}}(y\mid x)
\exp\!\left(\frac{r(x,y)}{\beta}\right).
{{< /katex >}}

The unrestricted KL-regularized optimum is

{{< katex display=true >}}
\boxed{\pi_r^*(y\mid x)
=\frac{\pi_{\mathrm{ref}}(y\mid x)}{Z_r(x)}
\exp\!\left(\frac{r(x,y)}{\beta}\right).}
{{< /katex >}}

{{% details "Deriving the optimal policy" %}}

For the per-prompt objective $J_x$, substitution yields

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

Nonnegativity of KL establishes optimality at $\pi=\pi_r^*$. A restricted neural policy class may not realize this distribution.

{{% /details %}}

For example, a uniform two-response reference with rewards $(\log 3,0)$ and $\beta=1$ yields optimal probabilities $(3/4,1/4)$.

### Eliminating the Unknown Normalizer

Inverting the optimal-policy relation gives

{{< katex display=true >}}
r(x,y)
=\beta\log\frac{\pi_r^*(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
+\beta\log Z_r(x).
{{< /katex >}}

The prompt-only term cancels in comparisons for the same prompt. Exploiting this offset invariance, define the implicit reward

{{< katex display=true >}}
\widehat r_\theta(x,y)
=\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}.
{{< /katex >}}

Its pairwise margin is

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

Substitution into the BT likelihood yields

{{< katex display=true >}}
\boxed{\mathcal{L}_{\mathrm{DPO}}(\theta)
=-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}_{\mathrm{pref}}}
\left[\log\sigma\!\left(m_\theta(x,y_w,y_l)\right)\right].}
{{< /katex >}}

## RLHF vs. DPO

Here **RLHF** denotes explicit reward estimation followed by policy optimization; **DPO** couples both through a policy-based reward parameterization.

### Equivalence Depends on What We Can Represent

A restricted policy class $\Pi$ induces a corresponding reward class:

{{< katex display=true >}}
\begin{aligned}
\widehat r_\pi(x,y)
&=\beta\log\frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}, \\
\mathcal{F}_{\Pi}
&=\left\{\widehat r_\pi+c:\pi\in\Pi,\ c:\mathcal{X}\to\mathbb{R}\right\}.
\end{aligned}
{{< /katex >}}

Here $\mathcal{X}$ is the prompt space and $c$ an arbitrary offset. Explicit reward learning instead selects a separate class $\mathcal{F}$.

[Shi et al. (2025, revised 2026)](https://arxiv.org/html/2505.19770v5) characterize the resulting gap under BT preferences, full-support reference sampling, and exact population optimization. The following compares true KL-regularized value at fixed $\Pi$, prompt distribution, reference, and $\beta$. Let $r^\star$ be the true reward and $\pi^\star$ its unrestricted optimum; reward realizability is modulo prompt offsets.

| Reward class contains $r^\star$? | Policy class contains $\pi^\star$? | Result |
| :---: | :---: | :--- |
| Yes | Yes | Both attain the optimum. |
| Yes | No | RLHF attains the best value within $\Pi$; DPO can be worse. |
| No | Yes | DPO attains the optimum; RLHF can be worse. |
| No | No | Neither method has a universal advantage. |

Under misspecification, maximizing regularized value and fitting preference likelihood can select different policies. Shi et al. additionally establish a finite-sample advantage for explicit reward estimation in a sparse-reward construction; the result is specific to that setting.

### Reward Parameterization Changes Generalization

Explicit and implicit reward models can use the same base model, dataset, and BT loss with different scoring rules:

{{< katex display=true >}}
\begin{aligned}
r_{\mathrm{explicit}}(x,y)&=w^\top h_\phi(x,y), \\
r_{\mathrm{implicit}}(x,y)&=\beta\log
\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}.
\end{aligned}
{{< /katex >}}

Here $h_\phi$ denotes the prompt–response representation.

[Razin et al. (2026)](https://arxiv.org/html/2507.07981v3) link implicit rewards to stronger reliance on surface tokens through frozen-representation theory and language-model experiments. Explicit heads are more robust to paraphrasing and translation in their evaluations; implicit models can match or outperform them under domain shifts. Accurate implicit verification also need not entail efficient generation.

These findings concern **reward-ranking generalization**, not universal superiority of either policy-learning method. Preference accuracy, generated-response quality, and training cost remain distinct evaluation criteria.

## Iterative Preference Learning

Fixed comparisons limit identifiability. If data compare only $a$ and $b$, rewards agreeing on these responses but differing on $c$ are observationally indistinguishable. Additional optimization cannot resolve this ambiguity without further assumptions or feedback.

### Iteration Changes the Available Evidence

[Xiong et al. (2024)](https://arxiv.org/html/2312.11456v4) distinguish offline learning without new oracle queries, online learning with new queries, and hybrid learning initialized from offline data. Their guarantees depend on specified statistical models and exploration oracles, which neural implementations approximate.

With fixed reference $\pi_0$, current policy $\pi_t$, and comparison policy $q_t$, a DPO-based iteration is:

1. Sample prompts, then responses from $\pi_t$ and $q_t$.
2. Query preferences to form a batch $\mathcal{B}_t$.
3. Accumulate the comparisons.
4. Refit the policy and repeat.

{{< katex display=true >}}
\begin{aligned}
y_a&\sim\pi_t(\cdot\mid x),\qquad y_b\sim q_t(\cdot\mid x), \\
\mathcal{D}_{t+1}&=\mathcal{D}_t\cup\mathcal{B}_t, \\
\pi_{t+1}&\in\arg\min_{\pi\in\Pi}
\mathcal{L}_{\mathrm{DPO}}\!\left(\pi;\pi_0,\mathcal{D}_{t+1}\right).
\end{aligned}
{{< /katex >}}

Their hybrid experiment (Appendix H.1) compares latest-policy and initial-policy responses, using UltraRM to simulate feedback rather than collecting human judgments. Each round restarts DPO from the initial model on accumulated data. Their offline multi-step rejection-sampling method uses proxy labels. Iterative querying can add evidence; additional epochs only reuse it.

### Sampling, Feedback, and the Reference Are Separate Choices

The policies $\pi_t$ and $q_t$ control data collection; $\pi_0$ defines regularization. Updating the former does not require resetting the latter. For a fixed reward $r$, exact distribution-level updates using the preceding policy as reference satisfy

{{< katex display=true >}}
\begin{aligned}
\pi_{t+1}(y\mid x)
&\propto\pi_t(y\mid x)\exp\!\left(\frac{r(x,y)}{\beta}\right), \\
\pi_T(y\mid x)
&\propto\pi_0(y\mid x)\exp\!\left(\frac{T\,r(x,y)}{\beta}\right).
\end{aligned}
{{< /katex >}}

Thus reference resets effectively reduce the original KL coefficient to $\beta/T$ under these assumptions. A fixed reference preserves the original objective.

Sampling coverage does not ensure feedback accuracy. A proxy fitted on existing comparisons can label new responses, but its predictions provide no independent evidence of their correctness. Iterative learning therefore requires both informative comparisons and reliable feedback; resampling alone guarantees neither.

## References

{{< references >}}
<li>Bradley, R. A., &amp; Terry, M. E. (1952). Rank analysis of incomplete block designs: I. The method of paired comparisons. <em>Biometrika</em>, 39(3–4), 324–345.</li>
<li>Ziegler, D. M., Stiennon, N., Wu, J., Brown, T. B., Radford, A., Amodei, D., Christiano, P., &amp; Irving, G. (2019). Fine-tuning language models from human preferences. arXiv:1909.08593.</li>
<li>Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., et al. (2022). Training language models to follow instructions with human feedback. <em>Advances in Neural Information Processing Systems</em>, 35, 27730–27744.</li>
<li>Schulman, J., Wolski, F., Dhariwal, P., Radford, A., &amp; Klimov, O. (2017). Proximal policy optimization algorithms. arXiv:1707.06347.</li>
<li>Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., &amp; Finn, C. (2023). Direct preference optimization: Your language model is secretly a reward model. <em>Advances in Neural Information Processing Systems</em>, 36, 53728–53741.</li>
<li>Shi, R., Song, M., Zhou, R., Zhang, Z., Fazel, M., &amp; Du, S. S. (2025; revised 2026). Understanding the performance gap in preference learning: A dichotomy of RLHF and DPO. arXiv:2505.19770.</li>
<li>Razin, N., Lin, Y., Yao, J., &amp; Arora, S. (2026). Why is your language model a poor implicit reward model? <em>International Conference on Learning Representations</em>. arXiv:2507.07981.</li>
<li>Xiong, W., Dong, H., Ye, C., Wang, Z., Zhong, H., Ji, H., Jiang, N., &amp; Zhang, T. (2024). Iterative preference learning from human feedback: Bridging theory and practice for RLHF under KL-constraint. <em>Proceedings of the International Conference on Machine Learning</em>. arXiv:2312.11456.</li>

{{< /references >}}
