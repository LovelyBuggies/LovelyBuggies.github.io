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
{{< badge style="black" title="License" value="CC BY-NC" >}}

RL is everywhere these days when people talk about LLMs. However, the challenge arises in how we design an appropriate reward model for evaluating task completions in languages (e.g., English, code, even math). This post walks through the existing practices of LLM reward modeling, including what’s working, what’s not, and why. It then takes a step back to ask whether today’s LLM reward models really make sense, and explores where the design should be heading.

## Human Languages

### Why We Read/Listen/Speak/Write?

This title might be overgeneralizing the usage of language into just 4 aspects (of course, there are many other ways we use language, like [thinking](https://lilianweng.github.io/posts/2025-05-01-thinking/)). Regardless,

<span class="text-danger"><strong>Why do we need language?</strong></span>

Language is an abstract and compact medium for expressing real-world dynamics, where vocabulary gives us the atomic units of meaning and syntax provides us the framework. With language, we can not only represent the accessible (visible, audible, tangible) signals of the world, but also the inaccessible parts (like atoms, gravity, microorganisms). This allows us to represent the world in a way that can be transmitted, reasoned about, and built upon. A good language system strives for compactness, mapping the universe as neatly as possible. But in practice, languages can rarely compress the world into pure states; they are normally noisy observations.

### Solving Tasks in Language

When people try to solve tasks in language, how to express them and evaluate the outcome are crucial questions. Some tasks, along with their completions, can be clearly conveyed using niche languages, e.g.,  asking a friend to bring back my phone from gym locker. For this kind of tasks, we can clearly express our intent and the outcome can be easily verified. But some other tasks may not have clear and objective evaluation criteria — deciding whether a paper (aka a research task) is "good" often lacks consensus (reflected in complaints about conference reviewing :3).

In fact, existing human languages are very limited in expressiveness. Just like I'm clumsy when describing a perfume (supposed that I have a static objective), my vocabulary for those is too few and vague to accurately express my true intent. We probably have a rich vocabulary for optical signals, but that for sound, touch, or scent is sparse, and we don't even design many words for electrical, magnetic and thermal signals. So it's very hard to accurately assign a task in those domains with limited words, no to mention evaluate it.

{{< sidenote >}}
<strong>Fun fact:</strong> This limitation may hint at why <a href="https://www.youtube.com/watch?v=fsvKLxmtFmY">LLMs are not the ultimate future of AI</a>. Based on humans' existing languages, it's likely that they will achieve human-like level of intelligence (<em>though it’s fun to know how much storage would be used to memorize our current knowledge base</em>). Look at how AlphaGo defeated Sedol Lee -- it doesn’t rely on language representations at all. But the optimistic thing is, we are still inventing new vocabularies and languages (e.g., <a href="https://go.dev/">Go</a> in 2007) to try to make breakthroughs. 
{{< /sidenote >}}

## RL Fine-Tuning

Recently, RL is one of  an important tool to fine tuning pretrained models to make them practical. As transformers scale to LMs, their raw outputs (optimized primarily for next-token prediction) often diverge from expected traits, so they need a secondary training phase to be specialized to certain domains. After initial SFT injects curated human-labeled data to the base transformers, a reward model (RM) is built to guide RL optimization.

{{< image src="/imgs/blog/reward_modeling_llm/RLHF.png" alt="RLHF" class="w-60" >}}

### RL Components in Language

<span class="text-danger"><strong>How do we reward the task completion in language?</strong></span>

As rewarding the tokens/words makes no sense semantically/syntactically in practice (and it's also computionally expensive), one may take it granted to use the prompts and responses as observations and actions (Shao et al. 2024). However, this assumption is not entirely sound. When the prompts and responses are long either from text-length perspective or dialogue-turn perspective, its obsure which parts actually contribute/hinder (i.e., credit assignment). So people are designing a bunch of tricks for reward modeling to migrate this issue as discussed later in the RLVR section.

### "Good" Justified by Humans

<span class="text-danger"><strong>How to create a reward model that aligns with human values?</strong></span>

Bradley-Terry (BT) model are usually used to build reward models for RLHF and can generalize preference signals to unseen inputs, scaling alignment by reducing reliance on slow and costly human annotations.

<div class="definition">
<strong>Definition 1:</strong> The original BT model posits that, given a pair of options $i$ and $j$ drawn from some population, the probability of selecting $i$ is,
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

Suppose a prompt $x$ is associated with $N$ candidate responses {{< katex >}} \{y_1, \ldots, y_N\} {{< /katex >}}, and human annotators provide preference labels between some pairs. Assuming human's annotation biases are trivial, tokenization and embedding are order-preserving, and sufficiently many comparisons $O(N \log N)$ are available under deterministic preferences, the true reward values can be inferred.

BT is **anti-symmetric**, the preference between two responses depends only on the difference in their reward values to ensure consistent and transitive pairwise comparisons. Inferring a reward model using the BT framework can be thus formulated as parameter estimation.

{{% details "Estimating Reward Parameters in BT and PL" %}}

Under BT,

{{< katex display=true >}}
\Pr(y_i \succ y_j \mid x)
= \frac{\exp(r(x,y_i))}{\exp(r(x,y_i)) + \exp(r(x,y_j))}\, .
{{< /katex >}}

Given $M$ annotated comparisons $\mathcal{C}=\{(i_m,j_m)\}_{m=1}^M$, the likelihood and log-likelihood are,

{{< katex display=true >}}
\mathcal{L}(r) = \prod_{m=1}^M \frac{\exp(r(x,y_{i_m}))}{\exp(r(x,y_{i_m})) + \exp(r(x,y_{j_m}))} .
{{< /katex >}}

Then the reward model can be estimated by MLE,

{{< katex display=true >}}
\begin{equation}
\begin{aligned}
\hat{r} &= \arg\max_r \log \mathcal{L}(r) \\
&=  \arg\max_r \sum_{m=1}^M \left[ r(x,y_{i_m}) - \log(\exp(r(x,y_{i_m})) + \exp(r(x,y_{j_m}))) \right] .\nonumber
\end{aligned}
\end{equation}
{{< /katex >}}

Similar, PL is also anti-symmetric, where swapping two responses in a ranking inverts the relative score difference in the likelihood. Given $M$ rankings $\mathcal{C} = \{(y_{i_1^m}, \ldots, y_{i_{N_m}^m})\}_{m=1}^M$,

{{< katex display=true >}}
\Pr(y_{i_1} \succ y_{i_2} \succ \ldots \succ y_{i_N} \mid x)
= \prod_{k=1}^{N-1} \frac{\exp(r(x, y_{i_k}))}{\sum_{j=k}^{N} \exp(r(x, y_{i_j}))}\, .
{{< /katex >}}

And the likelihood is,

{{< katex display=true >}}
\mathcal{L}(r) = \prod_{m=1}^{M} \prod_{k=1}^{N_m-1} \frac{\exp(r(x, y_{i_k^m}))}{\sum_{j=k}^{N_m} \exp(r(x, y_{i_j^m}))}.
{{< /katex >}}

Hence, by MLE, we have,

{{< katex display=true >}}
\begin{equation}
\begin{aligned}
\hat{r} &= \arg\max_r \log \mathcal{L}(r) \\
&= \arg\max_r \sum_{m=1}^{M} \sum_{k=1}^{N_m-1} \Big[ r(x, y_{i_k^m}) - \log\!\Big(\sum_{j=k}^{N_m} \exp(r(x, y_{i_j^m}))\Big) \Big].\nonumber
\end{aligned}
\end{equation}
{{< /katex >}}

{{% /details %}}

{{< sidenote >}}

<strong>Symmetric RM:</strong> In contrast, symmetric RM predicts the reward for each prompt–response pair independently. Modeling rewards using ground-truth symmetric signals is straightforward with traditional machine learning techniques, like regression model or binary classifier with cross-entropy loss. The main challenge for these methods lies in obtaining a sufficient amount of such data.

{{< /sidenote >}}


### "Good" as Verified

Recent advanced LLMs, such as o3-mini, have achieved performance comparable to human experts in domains like Olympic math (Balunović et al., 2025). In this case,

<span class="text-danger"><strong>Does the general human values really matter?</strong></span>

Admittedly, human attempts to construct values aim to reflect the truth, but they are never quite identical to it. Using verified signals from deterministic tools (verifiers or rules) to train LLMs can make them more objective and less biased, aka RLVR.

However, a common challenge in applying RLVR is that sparsity of outcome-based rewards. Given a dictionary, in theory, one could write *Hamlet*, yet in practice this is extremely difficult. Although other RL training setups share the same problem, it is more severe in LLM training since the language representation spaces are much larger (and exponentially in turn). 

**Horizontally**, people attempt to design some appropriate verifiable rubrics to shape the reward, e.g., multi-dimensional rewards (Lifshitz et al., 2025). Besides simple weighted sums, one can also give partial rewards for good process (Uesato et al, 2022), or prioritize some fundamental rubrics in a hierarchical manner (Lai et. al, 2024). However, it remains unclear what constitutes good rubrics, what should be the relation between them, and whether the shaping shifts the optimal policies. Moreover, if in a in single-turn training settings, the rubrics may still be too sparse.

To make RM denser **vertically**, one way is to fine-tune it in separate phases. When the raw outputs of base transformers preserve language structure but are not directly useful, models can be further trained in an additional phase, akin to offline RL. Multi-phase training can be useful when environments are not always static, yet in each phase, the training still aims only to optimize toward a fixed ground truth. Even when child models are diversified hierarchically from the base and specialized for particular tasks, the approach remains inefficient and lacks generality. Moreover, when the horizon is long, the training achieved in earlier phases becomes fragile and prone to degradation. Instead of having a sea of specialized models for different scenarios, we want agents to interact with their environment (and with external agents) to discover the optimal solutions.

## Rewarding Multi-Turn Dialogue

By interacting with external tools or models (should be stationary), agents can obtain intermediate feedback signals, which may be explicit in the next-turn prompt or implicit in the form of turn-level rewards. These feedbacks enable agents to enhance previous imperfectnesses and gradually develop policies. When making rewards singnal denser vertically, multi-phase, multi-turn training presents a trade-off — in general, the fewer phases the model goes through, the more general its behavior tends to remain.

{{< sidenote >}}
<strong>Example:</strong> An coder agent is asked to write well-formatted code, but it doesn't know "what exactly should be a good format". The external feedback could from different static analyzers at each turn, e.g., <a href="https://black.readthedocs.io/en/stable/">black</a>, <a href="https://github.com/hhatto/autopep8">autopep</a>, or <a href="https://www.pylint.org/">pylint</a>. After sufficient fine-tuning, the optimal policies learned under these external agents would be obviously different. Ideally, we want a general agent to explore the formatting requirements by itself through several rounds of interaction, rather than having separate ones to satisfy each. 
{{< /sidenote >}}

{{< sidenote >}}
<strong> Fun fact:</strong> I encountered this problem myself. Both black and autopep8 were installed in my pre-commit hooks, but I let Claude Code to follow black, which led to formatting conflicts when committing code.
{{< /sidenote >}}

### Magic of Positivity

In RL training, people are careful when making the rewards for expected outcome larger than less expected ones when designing rewards. But, the positivity of rewards are usually overlooked, or defaultly set to be non-negative.

<span class="text-danger"><strong>Does the reward positivity matters?</strong></span>

<span><strong>Yes, it does!</strong></span> Suppose an agent is training in an episodic case, which either terminates by reaching horizon limit or triggers certain conditions, non-negative rewards ($\forall r, r \geqslant 0$) encourage the agent to stay longer in the environment to explore potential benefits ("don't push me away"), since it doesn't hurt anyway; non-positive rewards ($\forall r, r \leqslant 0$), on the contrary, makes the agent suffer and want to escape asap ("let me go"). 

Given non-negative rewards ($r \geqslant 0$), agents are naturally encouraged to strive longer before eventual doom. However, pursuing a positive objective may be delayed or even become risky. Of course, one can strengthen the terminal outcomes by shaping them with dominant bonuses or penalties, as in SMAC (Samvelyan et al., 2019) or OverCook (Gessler et al., 2025). Yet, even with such shaping, the learned policy often remains suboptimal. For example, once a victory is secured (win the game or kill an enemy), there is no need to expose agents to additional risk simply to casue more damage.

{{< sidenote >}}
<strong>Clarification:</strong> This doesn't say SMAC's design is incorrect, The design of this environment is tricky :D All enemy agents must reach HP = 0 for the game to count as a win, the total amount of damage required is hence fixed, so there is no room to use less damage to win the game. But what remains optimizable is how you allocate that damage. For example, if an enemy unit is weak with only 1 HP left, it’s wasteful and risky to send in a powerful but fragile attacker. A smarter choice is to let a healthy or more defensive unit finish the kill.
{{< /sidenote >}}

Non-positive rewards ($\leqslant 0$) are used less often, since designers typically wish to avoid letting agents to surrender prematurely. However, it makes more sense when the termination is negative. If the termination is positive, such as completing a task, non-positive rewards can be beneficial, as they push agents to solve the task quickly rather than prolonging interactions. It may be overly harsh to impose a large penalty every time the task is not completed, introducing a discount factor can be useful in this case. Therefore, in LLM training, a useful practice is to employ discounted negative rewards, maximizing the return will encourage agents to complete the task faster.

In practice, a hybrid scheme can be applied by combining a negative lower limit with a positive upper limit. When the agent achieves progress, we can signal encouragement — "it gets something and has some potential, let's explore more". Conversely, when the agent fails to achieve the objective, it is often better to cut losses in time, preventing wasted effort on unpromising trajectories.

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
<li>Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., ... & Anadkat, S. (2024). Gpt-4 technical report. arXiv 2023. arXiv preprint arXiv:2303.08774.</li>
<li>Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., ... & Guo, D. (2024). Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300. </li>
<li>Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., ... & Scialom, T. (2023). Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.</li>
<li>Sun, H., Shen, Y., & Ton, J. F. (2024). Rethinking bradley-terry models in preference-based reward modeling: Foundations, theory, and alternatives. arXiv preprint arXiv:2411.04991.</li>
<li>Weng, L. (2024). “Reward Hacking in Reinforcement Learning.”</li>
<li>Balunović, M., Dekoninck, J., Petrov, I., Jovanović, N., & Vechev, M. (2025). Matharena: Evaluating llms on uncontaminated math competitions. arXiv preprint arXiv:2505.23281.</li>
<li>Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., ... & He, Y. (2025). Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948.</li>
<li>Lifshitz, S., McIlraith, S. A., & Du, Y. (2025). Multi-agent verification: Scaling test-time compute with multiple verifiers. arXiv preprint arXiv:2502.20379.</li>
<li>Lai, Y., Wang, S., Liu, S., Huang, X., & Wei, Z. (2024). ALaRM: Align language models via hierarchical rewards modeling. arXiv preprint arXiv:2403.06754.</li>
<li>Uesato, J., Kushman, N., Kumar, R., Song, F., Siegel, N., Wang, L., ... & Higgins, I. (2022). Solving math word problems with process-and outcome-based feedback. arXiv preprint arXiv:2211.14275.</li>
<li>Samvelyan, M., Rashid, T., De Witt, C. S., Farquhar, G., Nardelli, N., Rudner, T. G., ... & Whiteson, S. (2019). The starcraft multi-agent challenge. arXiv preprint arXiv:1902.04043.</li>
<li>Gessler, T., Dizdarevic, T., Calinescu, A., Ellis, B., Lupu, A., & Foerster, J. N. (2025). Overcookedv2: Rethinking overcooked for zero-shot coordination. arXiv preprint arXiv:2503.17821.</li>

{{< /references >}}
