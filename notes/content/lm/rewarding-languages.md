---
date: 2025-09-18
title: "All You Want to Know about LLM Rewards"
math: true
postType: thought
linkTitle: "Rewarding Languages"
readingTime: 50
---

{{< katex />}}

# All You Want to Know about LLM Rewards
{{< postbadges >}}
{{< badge style="black" title="License" value="CC By-NC" >}}

RL is everywhere these days when people talk about LLMs. However, the challenge arises in how we design an appropriate reward model for evaluating task completions in languages (e.g., English, code, even math). This post walks through the existing practices of LLM reward modeling, including what’s working, what’s not, and why. It then takes a step back to ask whether today’s LLM reward models really make sense, and explores where the design should be heading.

## Human Languages

### Why We Read/Listen/Speak/Write?

This title might be overgeneralizing the usage of language into just 4 aspects (of course, there are many other ways we use language, like [thinking](https://lilianweng.github.io/posts/2025-05-01-thinking/)). Regardless,

<span class="text-danger"><strong>Why do humans need language?</strong></span>

Language is an abstract and compact medium for expressing real-world dynamics, where vocabulary gives us the atomic units of meaning and syntax provides us the framework. With language, we can not only represent the accessible (visible, audible, tangible) signals of the world, but also the inaccessible parts (like atoms, gravity, microorganisms). This allows us to represent the world in a way that can be transmitted, reasoned about, and built upon. A good language system strives for compactness, mapping the universe as neatly as possible. But in practice, languages can rarely compress the world into pure states; they are normally noisy observations.

### Solving Tasks in Language

When people try to solve tasks in language, how to express them and evaluate the outcome are crucial questions. Some tasks, along with their completions, can be clearly conveyed using niche languages, e.g.,  asking a friend to bring back my phone from gym locker. For this task, we can clearly express our intent and verify the result by observing it. But some other tasks don’t have clear and objective evaluation criteria, like deciding whether a paper (aka a research task) is "good" often lacks consensus (reflected in complaints about conference reviewing :3).

In fact, existing human languages are very limited in expressiveness. Just like my dilemma when describing a perfume or a haircut (supposed that I have a static objective), my vocabulary for those is too few and vague to accurately express my intent. We probably have a rich vocabulary for optical signals, but that for sound, touch, or scent is sparse. And we don't even design many words for electrical, magnetic and thermal signals. These tasks related are hard to accurately represent by language, and thus unable to assign and evaluate.

{{< sidenote >}}
This limitation may hint at why <a href="https://www.youtube.com/watch?v=fsvKLxmtFmY">LLMs are not the ultimate future of AI</a>. Based on humans' existing languages, they can only achieve human-like level of intelligence (<em>though it’s fun to know how much storage would be used to memorize our current knowledge base</em>). Look at how AlphaGo defeated Sedol Lee -- it doesn’t rely on language representations at all. But the optimistic thing is, we are still inventing new vocabularies and languages (e.g., <a href="https://go.dev/">Go</a> in 2007) to try to make breakthroughs. 
{{< /sidenote >}}

## RL Fine-Tuning

Recently, RL is one of  an important tool to fine tuning pretrained models to make them practical. As transformers scale to LMs, their raw outputs (optimized primarily for next-token prediction) often diverge from expected traits, so they need a secondary training phase to be specialized to certain domains. Normally, this phase involves supervised fine-tuning (SFT), reward modeling, and RL. After initial SFT injects curated human-labeled data to the base transformers, a reward model is built, either based on external rules or human preference. While it only serves as a partial approximation of the ultimate evaluation, the reward model plays a crucial role in guiding optimization and is thus crucial for the training.

So, <span class="text-danger"><strong>how do we reward the task completion in human languages?</strong></span>

{{< image src="/imgs/blog/reward_modeling_llm/RLHF.png" alt="RLHF" class="w-60" >}}

## RLHF: "Good" as Justified by Humans

### Anti-Symmetric Preference Modeling

Reward models can be trainable proxies for human preference. This kinds of reward models are usually built based on Bradley–Terry (BT) model and can generalize preference signals to unseen inputs, scaling alignment by reducing reliance on slow and costly human annotations.

<div class="definition">
<strong>Definition 1:</strong> The original BT model posits that, given a pair of options $i$ and $j$ drawn from some population, the probability of selecting $i$ is

{{< katex display=true >}}
\Pr(i \succ j) = \frac{u_i}{u_i + u_j}
=\frac{\exp(r(i))}{\exp(r(i)) + \exp(r(j))}\, ,\label{eq:bt}
{{< /katex >}}

where $u_i$ and $u_j$ are the respective utility or preference of options $i$ and $j$, commonly parameterized in the exponential form via rewards $r(i)$ and $r(j)$. It can be further extended to rank $N$ options, known as the Plackett–Luce (PL) model,

{{< katex display=true >}}
\Pr(i_1 \succ i_2 \succ \cdots \succ i_N)
= \prod_{k=1}^{N} \frac{\exp(r(i_k))}{\sum_{j=k}^{N} \exp(r(i_j))}\, .\label{eq:pl}
{{< /katex >}}

</div>

BT is **anti-symmetric**, the preference between two responses depends only on the difference in their reward values. This structure ensures consistent and transitive pairwise comparisons. Inferring a reward model using the BT framework can be thus formulated as parameter estimation: recover latent reward values for candidate responses based on observed pairwise comparisons.

Suppose a prompt $x$ is associated with $N$ candidate responses {{< katex >}} \{y_1, \ldots, y_N\} {{< /katex >}}, and human annotators provide preference labels between some pairs. Assuming human's annotation biases are trivial, tokenization and embedding are order-preserving, and sufficiently many comparisons $O(N \log N)$ are available under deterministic preferences, the true reward values can be inferred.

Let each observed preference be a pair $(i \succ j)$ indicating that $y_i$ is preferred over $y_j$ for prompt $x$. Under BT,

{{< katex display=true >}}
\Pr(y_i \succ y_j \mid x)
= \frac{\exp(r(x,y_i))}{\exp(r(x,y_i)) + \exp(r(x,y_j))}\, .
{{< /katex >}}

Given $M$ annotated comparisons $\mathcal{C}=\{(i_m,j_m)\}_{m=1}^M$, the likelihood and log-likelihood are

{{< katex display=true >}}
\mathcal{L}(r) = \prod_{m=1}^M \frac{\exp(r(x,y_{i_m}))}{\exp(r(x,y_{i_m})) + \exp(r(x,y_{j_m}))}\, ,
\\
\log \mathcal{L}(r) = \sum_{m=1}^M \Big[ r(x,y_{i_m}) - \log(\exp(r(x,y_{i_m})) + \exp(r(x,y_{j_m}))) \Big] \, .
{{< /katex >}}

The reward model can be estimated by MLE,

$$r^\* = \arg\max_r \log \mathcal{L}(r).$$

#### Reward Modeling with Ranked Preferences (PL Model)

While BT uses pairwise preferences, real systems can collect ranked lists. For a ranking $(y_{i_1} \succ y_{i_2} \succ \ldots \succ y_{i_N})$ for prompt $x$, the PL probability is

{{< katex display=true >}}
\Pr(y_{i_1} \succ y_{i_2} \succ \ldots \succ y_{i_N} \mid x)
= \prod_{k=1}^{N-1} \frac{\exp(r(x, y_{i_k}))}{\sum_{j=k}^{N} \exp(r(x, y_{i_j}))}\, .
{{< /katex >}}

Given $M$ rankings $\mathcal{C} = \{(y_{i_1^m}, \ldots, y_{i_{N_m}^m})\}_{m=1}^M$, the likelihood and log-likelihood are

{{< katex display=true >}}
\mathcal{L}(r) = \prod_{m=1}^{M} \prod_{k=1}^{N_m-1} \frac{\exp(r(x, y_{i_k^m}))}{\sum_{j=k}^{N_m} \exp(r(x, y_{i_j^m}))}\, ,\\
\log \mathcal{L}(r) = \sum_{m=1}^{M} \sum_{k=1}^{N_m-1} \Big[ r(x, y_{i_k^m}) - \log\!\Big(\sum_{j=k}^{N_m} \exp(r(x, y_{i_j^m}))\Big) \Big] \, ,
{{< /katex >}}

and the MLE $\; r^* = \arg\max_r \log \mathcal{L}(r)$. Preference modeling with PL is also anti-symmetric: swapping two responses in a ranking inverts the relative score difference in the likelihood.

### Symmetric Reward Modeling

As cliché as it sounds, modeling rewards through symmetric signals is feasible. Symmetric models predict the reward for each prompt–response pair independently, without referencing alternatives.

#### Regression-based Reward Model

Given scalar human ratings $\{(x_n, y_n, s_n)\}_{n=1}^N$ with $s_n\in\mathbb{R}$, train $r(x,y)$ by minimizing

{{< katex display=true >}}
\mathcal{L}_{\text{reg}} = \sum_{n=1}^N \big(r(x_n, y_n) - s_n\big)^2\, .
{{< /katex >}}

Such data appears in datasets like Anthropic HH, OpenAssistant, and MT-Bench (numeric quality scores per response).

#### Classification-based Reward Model

Alternatively, train a binary classifier using labels $s_n\in\{0,1\}$ for acceptability, with sigmoid $\sigma$ and cross-entropy loss

{{< katex display=true >}}
\mathcal{L}_{\text{cls}} = - \sum_{n=1}^N \Big[ s_n \log \sigma(r(x_n, y_n)) + (1 - s_n) \log(1 - \sigma(r(x_n, y_n))) \Big] \, .
{{< /katex >}}

### Other Reward Modeling Techniques

#### Inverse Reinforcement Learning

IRL aims to recover a reward function explaining expert behavior in an MDP. Let $(\mathcal{X}, \mathcal{Y}, T, \gamma)$ be the MDP and expert trajectories $\{\tau_i\}$ where $\tau_i=(x_0,y_0,x_1,y_1,\dots)$. Infer $r: \mathcal{X}\times\mathcal{Y}\to\mathbb{R}$ such that the induced optimal policy matches observed behavior. However, for LLM alignment this mismatches due to: (i) unstructured, high-dimensional language; (ii) feedback as relative preferences rather than optimal sequences; (iii) heavy compute (re-solving RL repeatedly) impractical at LLM scale.

#### Bayesian Reward Learning

Maintain a posterior over reward parameters $p(\theta\mid D) \propto p(D\mid\theta) p(\theta)$ to represent uncertainty and enable posterior-aware policies. In high-dimensional reward spaces, exact/accurate inference is expensive and often impractical for large LLMs.


## RLVR: "Good" as Verified by Analyzers

too sparse

## Finer-Granular Rewards

reward is too sparse

horizontally: reward shaping, but that's still one turn

vertically: multi-turn provides intermediate reward signals

## Problems of Current Reward Models

not multi-turn rewards

## Citation

<div class="cite-block">
{{< tabs >}}

{{% tab "Plain" %}}
```tpl
Liu, Shuo. (September 2025). All You Want to Know about LLM Rewards.
LovelyBuggies's Blog. https://lovelybuggies.github.io/notes/lm/rewarding-languages/
```
{{% /tab %}}

{{% tab "BibTeX" %}}
```bibtex
@article{liu2025allyouwanttoknowaboutllmrewards,
  title   = {All You Want to Know about LLM Rewards},
  author  = {Liu, Shuo},
  journal = {lovelybuggies.github.io},
  year    = {2025},
  month   = {September},
  url     = {https://lovelybuggies.github.io/notes/lm/rewarding-languages/}
}
```
{{% /tab %}}

{{< /tabs >}}
</div>

## References

{{< references >}}
<li>Achiam, Josh, Steven Adler, and Sandhini Agarwal. 2024. “GPT-4 Technical Report.” <https://arxiv.org/abs/2303.08774>.</li>
<li>Shao, Zhihong, Peiyi Wang, and Qihao Zhu. 2024. “DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.” <https://arxiv.org/abs/2402.03300>.</li>
<li>Touvron, Hugo, Louis Martin, and Kevin Stone. 2023. “Llama 2: Open Foundation and Fine-Tuned Chat Models.” <https://arxiv.org/abs/2307.09288>.</li>
<li>Sun, Zihan, Yixin Chen, Zexuan Feng, et al. 2025. “Rethinking Bradley–Terry Models for Preference-Based Reward Modeling.”</li>
<li>Weng, Lilian. 2024. “Reward Hacking in Reinforcement Learning.”</li>
{{< /references >}}


<!-- moved to root content -->
<!-- moved back under lm/ -->
