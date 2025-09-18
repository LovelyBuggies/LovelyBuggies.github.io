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
{{< badge style="black" title="Liscense" value="CC By-NC" >}}

RL is everywhere these days when people talk about LLMs. However, the challenge arises in how we design an appropriate reward model  for evaluating task completions in languages, (e.g., English, code, even math). This post walks through the existing practices of LLM reward modeling, including what’s working, what’s not, and why. It then takes a step back to ask whether today’s reward models really make sense, and explores where the design of next-generation LLM rewards might be heading.

## Human Languages

### Why We Read/Listen/Speak/Write?

This title might be overgeneralizing the usage of language into just 4 aspects (of course, there are many other ways we use language, like [thinking](https://lilianweng.github.io/posts/2025-05-01-thinking/)). But, basically,

**Why do we humans use language?**

Language is an abstract and compact medium for express real-world dynamics, where vocabulary gives us the atomic units of meaning and syntax provides us the framework. With the language, we can not only represent the accessible (visible, audible, tangible) signals of the world, but also the inaccessible parts (like atoms, gravity, microorganisms). This allows us to represent the world in a way that can be transmitted, reasoned about, and built upon. A good language system strives for compactness, mapping the universe as neatly as possible. But in practice, languages can rarely compress the world into pure states, they are normally noisy observations.

### Solving Tasks in Languages

When people try to solve tasks in languages, how to express them and evaluate the outcome are the crucial questions. Some tasks, along with their completions, can be clearly conveyed using niche languages. Supposed we are asking a friend to bring us an iPhone 17 from Apple Store. For this task, we can clearly express our intent and verify the result by observing it. But some other tasks don’t have clear and objective evaluation criteria, like deciding whether a paper (aka a research task) is "good" often lacks consensus (reflected in complaints about conference reviewing :3).

In fact, existing human languages are very limited in expressiveness. Although we have a rich vocabulary for describing vision signals, that for sound, touch, or scen is sparse. Just like my dilemma when describing perfumes or a haircut (supposed that I have a clear objective in my mind), my vocabulary is too few and vague to accurately express my intented outcome. For the tasks that are not able to be accurately represented by language, how can we expect them to be evaluated?

{{< sidenote >}}
This limitation may hint at why <a href="https://www.youtube.com/watch?v=fsvKLxmtFmY">LLMs are not the ultimate future of AI</a>. Based on human's existing languages, they can only achieve human-like level of intelligence (<em>though it’s fun to know how much storage would be use to memorize our current knowledge base</em>). Think about how AlphaGo defeated Sedol Lee -- it didn’t rely on language representations at all. But the optimistic thing is, we are still inventing new vocabularies and even entire new languages (e.g., <a href="https://go.dev/">Go</a> in 2007) to make breakthroughs. 
{{< /sidenote >}}

## RL Fine-Tuning

As LMs scale, their raw outputs (optimized primarily for next-token prediction) often diverge from expected traits. To adapt them to specific domains, a secondary fine-tuning phase is typically applied. A standard alignment pipeline involves 3 stages: supervised fine-tuning (SFT), reward modeling (RM), and RL fine-tuning. After initial SFT on a base transformer with curated human-labeled data, a reward model is built, either from explicit rules or human preference data. While only serves as an approximation of ultimate evaluation, the reward model plays a crucial role in guiding optimization and thus extremely important.

![RLHF](/imgs/blog/reward_modeling_llm/RLHF.png)

This leads to the core topic of this post: 

**How do we reward the task completion in human languages?**

## "Good"as  Justified by Human

### Anti-Symmetric Preference Modeling

In this section, we introduce the mainstream methods that model rewards of LLM responses through preference comparison.

#### BT Model and Its Ranking Extension

The original Bradley–Terry (BT) model posits that, given a pair of options $i$ and $j$ drawn from some population, the probability of selecting $i$ is

{{< katex display=true >}}
\Pr(i \succ j) = \frac{u_i}{u_i + u_j}
=\frac{\exp(r(i))}{\exp(r(i)) + \exp(r(j))}\, ,\label{eq:bt}
{{< /katex >}}

where $u_i$ and $u_j$ are the respective utility or preference of options $i$ and $j$, commonly parameterized in the exponential form via rewards $r(i)$ and $r(j)$. It can be further extended to rank $N$ options, known as the Plackett–Luce (PL) model,

{{< katex display=true >}}
\Pr(i_1 \succ i_2 \succ \cdots \succ i_N)
= \prod_{k=1}^{N} \frac{\exp(r(i_k))}{\sum_{j=k}^{N} \exp(r(i_j))}\, .\label{eq:pl}
{{< /katex >}}

BT is anti-symmetric: the preference between two responses depends only on the difference in their reward values. It satisfies $\Pr(y_i \succ y_j) = 1 - \Pr(y_j \succ y_i)$, and the log-odds of preference is anti-symmetric: $\log \!\left( \frac{\Pr(y_i \succ y_j)}{\Pr(y_j \succ y_i)} \right) = r(x, y_i) - r(x, y_j)$. This structure ensures consistent and transitive pairwise comparisons, making BT suitable for preference modeling (initially used to rank sports teams and players, e.g., Elo rating).

#### Reward Modeling with Pairwise Preferences (BT Model)

Inferring a reward model using the BT framework can be formulated as parameter estimation: recover latent reward values for candidate responses based on observed pairwise comparisons.

Suppose a prompt $x$ is associated with $N$ candidate responses $\{y_1, \ldots, y_N\}$, and human annotators provide preference labels between some pairs. Ideally, given sufficiently many comparisons under deterministic preference, the true reward values can be accurately inferred.\footnote{$O(N \log N)$ comparisons are sufficient for modeling rewards of $N$ responses.} In practice, however, this inference is challenged by stochastic human behavior and sparse annotation.

- Modeling assumptions (adapted from Sun et al., 2025):
  1. Deterministic responses and rewards: for a given prompt $x$, a response $y$ is deterministically generated by a model. The oracle reward $r(x,y)$ for each $(x,y)$ is fixed.
  2. Deterministic annotators with bounded bias: when an annotator $A$ compares responses, their preference is a deterministic comparison of biased reward evaluations,

{{< katex display=true >}}
\mathbf{1}\!\left( \underbrace{y_i \succ y_j}_{\text{decision}} \mid x, A \right)
= \mathbf{1}\!\left( \underbrace{r(x,y_i) + b(x,y_i;A) > r(x,y_j) + b(x,y_j;A)}_{\text{biased preference}} \right)\! .
{{< /katex >}}

  3. Order-preserving shaping: a known embedding $\Psi$ maps each $(x,y)$ to a feature space, constrained to be order-preserving and not affect optimization; high-dimensional embeddings can help generalization but may introduce reward hacking (see Section “Takeaways”).
  4. Imperfect human annotations: the annotator function $h(x_1,x_2,y_1,y_2)$ provides feedback that increasingly aligns with the oracle reward as the reward gap grows,

{{< katex display=true >}}
\mathbb{P}\!\left( h(x_1,x_2,y_1,y_2)\, (r(x_1,y_1) - r(x_2,y_2)) > 0 \;\middle|\; \Delta r \right) = \xi(\Delta r)\, ,
{{< /katex >}}

where $\Delta r := |r(x_1,y_1) - r(x_2,y_2)|$ and $\xi:[0,\infty)\to[0.5,1]$ is monotone increasing.

Likelihood and estimation. Let each observed preference be a pair $(i \succ j)$ indicating that $y_i$ is preferred over $y_j$ for prompt $x$. Under BT,

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

The optimal reward model is obtained by maximizing the log-likelihood (MLE), $\; r^* = \arg\max_r \log \mathcal{L}(r)$, identifiable up to an additive constant and consistent under the assumptions above.

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


## "Good" as Verified by Analyzer

too sparese

## Finer-Granular Rewards

reward is too sparse

horizontally: reward shaping, but that's still one turn

vertically: multi-turn provide intermediate reward signals

## Problems of Current Reward Models

not multi-turn rewards

## Citation

<div class="cite-block">
{{< tabs >}}

{{% tab "Plain" %}}
```text
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
