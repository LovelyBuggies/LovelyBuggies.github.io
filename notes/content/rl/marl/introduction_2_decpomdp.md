---
date: 2024-12-21
title: "Introduction to Dec-POMDP"
math: true
postType: notes
linkTitle: "Introduction to Dec-POMDP"
readingTime: 5
---

{{< katex />}}

# An Introduction to Dec-POMDP
{{< postbadges >}}

## Dec-POMDP

<div class="definition">
**Definition 1**. A Dec-POMDP is a tuple {{< katex >}}\langle \mathbb{I}, \mathcal{S}, \{\mathbb{A}_i\}, T, R, \{\mathbb{O}_i\}, O, \mathcal{H}, \gamma\rangle{{< /katex >}}:

- {{< katex >}}\mathbb{I}{{< /katex >}} is a finite sets of agents, {{< katex >}}|\mathbb{I}|=n{{< /katex >}};

- {{< katex >}}\mathcal{S}{{< /katex >}} is a set of states with designated initial state distribution {{< katex >}}b^0{{< /katex >}};

- {{< katex >}}\mathbb{A}_i{{< /katex >}} is a set of actions for agent {{< katex >}}i{{< /katex >}} with {{< katex >}}\mathbb{A}\doteq \times_i \mathbb{A}_i{{< /katex >}} the set of joint actions;

- {{< katex >}}T{{< /katex >}} is the state transition probability function, {{< katex >}}T{{< /katex >}}: {{< katex >}}\mathcal{S} \times \mathbb{A} \times \mathcal{S} \rightarrow [0, 1]{{< /katex >}}, that specifies the probability of transitioning from state {{< katex >}}s \in \mathcal{S}{{< /katex >}} to {{< katex >}}s' \in \mathcal{S}{{< /katex >}} when the actions {{< katex >}}\boldsymbol{a} \in \mathbb{A}{{< /katex >}} are taken by agents (i.e., {{< katex >}}T(s, \boldsymbol{a}, s')=P(s'|s, \textbf{a}){{< /katex >}});

- {{< katex >}}R{{< /katex >}} is the joint reward function, where {{< katex >}}R{{< /katex >}}: {{< katex >}}\mathcal{S} \times \mathbb{A} \rightarrow \mathbb{R}{{< /katex >}};

- {{< katex >}}\mathbb{O}_i{{< /katex >}} is a set of observations for each agent {{< katex >}}i{{< /katex >}}, with {{< katex >}}\mathbb{O}\doteq\times_i \mathbb{O}_i{{< /katex >}} the set of joint observations;

- {{< katex >}}O{{< /katex >}} is an observation probability function, {{< katex >}}O{{< /katex >}}: {{< katex >}}\mathbb{A} \times \mathcal{S} \times \mathbb{O}{{< /katex >}}, that specifies the probability of seeing observation {{< katex >}}\boldsymbol{o}' \in \mathbb{O}{{< /katex >}} given the actions {{< katex >}}\boldsymbol{a} \in \mathbb{A}{{< /katex >}} are taken and state {{< katex >}}s' \in \mathcal{S}{{< /katex >}} is observed (i.e., {{< katex >}}O(\boldsymbol{a}, s', \boldsymbol{o}')=P(\boldsymbol{o}'|\textbf{a},s'){{< /katex >}});

- {{< katex >}}\mathcal{H}{{< /katex >}} is the horizon (the number of steps until termination);

- {{< katex >}}\gamma{{< /katex >}} is the discount factor for the return.

A solution to a Dec-POMDP is a joint policy $\boldsymbol{\pi}:\mathbb{H}_i\to\mathbb{A}_i, \forall i \in \mathbb{I}$ over joint observation-action history $\boldsymbol{h}=\{\boldsymbol{a}^{0}, \boldsymbol{o}^{1}, \cdots \boldsymbol{o}^{\mathcal{H}-1}\}$, an optimal solution maximizes the expected return, {{< katex display=true >}}
\boldsymbol{\pi}^*=\mathop{\mathrm{argmax}}_{\boldsymbol{\pi}}\mathbb{E}\left[\textstyle\sum_{t=0}^{\mathcal{H}-1}R(\boldsymbol{h}, \boldsymbol{\pi}(\boldsymbol{h}))\middle|b^0\right].
{{< /katex >}}

</div>

In Dec-POMDP, the Bellman recursive formulation of the history V-function is,

{{< katex display=true >}}
\label{eq:decpomdp-V}
\begin{aligned}
    V^{\boldsymbol{\pi}}(\boldsymbol{h}) &= \sum_{s} P(s|b^0, \boldsymbol{h})\left[R(s, \boldsymbol{\pi}(\boldsymbol{h}))+\gamma \sum_{s'}P(s'|s, \boldsymbol{\pi}(\boldsymbol{h})) \sum_{\boldsymbol{o}'}P(\boldsymbol{o}'|\boldsymbol{\pi}(\boldsymbol{h}), s') V^{\boldsymbol{\pi}}(\boldsymbol{h}') \right]\\
    &\equiv R(\boldsymbol{h}, \boldsymbol{\pi})+\gamma\sum_{\boldsymbol{o}'}P(\boldsymbol{o}'|\boldsymbol{h}, \boldsymbol{\pi}) V^{\boldsymbol{\pi}}(\boldsymbol{h}'),
\end{aligned}
{{< /katex >}}
The definition of the value function is flexible: it may be based on the value of a state, a belief state, an observation, a state history, an observation history, a single action (single-step policy), a full policy (action history), observation–action history, or combinations of these.

the Bellman recursive formulation of the **history-policy** Q-function is,

{{< katex display=true >}}
\label{eq:decpomdp-q}
\small
\begin{aligned}
    Q^{\boldsymbol{\pi}}(\boldsymbol{h}, \boldsymbol{\pi}) &= \sum_{s} P(s|b^0, \boldsymbol{h})\left[R(s, \boldsymbol{\pi}(\boldsymbol{h}))+\gamma\sum_{s'}P(s'|s, \boldsymbol{\pi}(\boldsymbol{h})) \sum_{\boldsymbol{o}'}P(\boldsymbol{o}'|\boldsymbol{\pi}(\boldsymbol{h}), s') Q^{\boldsymbol{\pi}}(\boldsymbol{h}', \boldsymbol{\pi}(\boldsymbol{h}')) \right]\\
    &\equiv R(\boldsymbol{h}, \boldsymbol{\pi})+\gamma\sum_{\boldsymbol{o}'}P(\boldsymbol{o}'|\boldsymbol{h}, \boldsymbol{\pi}) Q^{\boldsymbol{\pi}}(\boldsymbol{h}', \boldsymbol{\pi}) .
\end{aligned}
{{< /katex >}}

## Dec-POMDP Subclasses

- **Centralized:** MMDP is a fully observable version of Dec-POMDP, but it does not specify decentralized control. Dec-MDP assumes that the joint observations uniquely determine the state, while agents still act with local observations. Similarly, MPOMDP does not specify whether the control is decentralized, which could have a centralized policy {{< katex >}}\mathbb{H}\to\mathbb{A}{{< /katex >}}.
- **Decentralized:** A decentralized control model might be factorized with independent local variables, e.g., transition-independence (TI) {{< katex >}}T(s, \boldsymbol{a}, s')=\Pi_{i=1}^{n} T(s_i, a_i, s_i'){{< /katex >}} and reward-independence (RI) {{< katex >}}R(s,\boldsymbol{\pi})=f_\text{mono}(\langle R(s_i, \pi_i)\rangle_{i=1}^{n}){{< /katex >}}. Network-distributed POMDP (ND-POMDP) represents the factored one with TI and block-RI, i.e., {{< katex >}}R(s,\boldsymbol{\pi})=f_\text{mono}(\langle R(s_{i, \mathcal{N}(i)}, \pi_{i, \mathcal{N}(i)})\rangle_{i=1}^{n}){{< /katex >}}, where {{< katex >}}{\mathcal{N}(i)}{{< /katex >}} are the neighbors of {{< katex >}}i{{< /katex >}}.

The **worst-case complexity** of finite-horizon problems is: (Amato et al., 2013)

| **Model**              | **Complexity**  |
|:-----------------------|:----------------|
| MDP                    | P-complete      |
| MMDP (Cen-MMDP)        | P-complete      |
| Dec-MDP                | NEXP-complete   |
| Dec-MDP with TI no RI  | NP-complete     |
| Dec-MDP with RI no TI  | NEXP-complete   |
| Dec-MDP with TI and RI | P-complete      |
| POMDP                  | PSPACE-complete |
| MPOMDP (Cen-MPOMDP)    | PSPACE-complete |
| Dec-POMDP              | NEXP-complete   |
| ND-POMDP               | NEXP-complete   |


{{% details title="Theorems" open=false %}}

<span id="tab:complexity" label="tab:complexity"></span>

<div class="theorem">
**Theorem 1**. An MDP is P-complete in finite and infinite horizons (Papadimitriou and Tsitsiklis 1987).

**Theorem 2**. A finite POMDP is PSPACE-complete (Papadimitriou and Tsitsiklis 1987).

**Theorem 3**. The complexity of an infinite POMDP is undecidable (Madani, Hanks, and Condon 1999), leading to the undecidability of the infinite Dec-POMDP complexity.

**Theorem 4**. A finite Dec-POMDP ($n\geqslant2$) is NEXP-complete, and a finite Dec-MDP ($_n\geqslant3$) is also NEXP-complete (Bernstein et al. 2002).

**Fact 1**. A Dec-MDP with TI and RI can be solved independently, resulting in P-complete.

**Theorem 5**. A Dec-MDP with TI and joint reward is NP-complete, a Dec-MDP with RI but no TI is NEXP-complete (Becker et al. 2004).

**Fact 2**. An ND-POMDP has the same worst-case complexity as a Dec-POMDP (Nair et al. 2005).

</div>

{{% /details %}}


## References

{{< references >}}
<li>Amato, Christopher, Girish Chowdhary, Alborz Geramifard, N. Kemal Üre, and Mykel J. Kochenderfer. 2013. “Decentralized Control of Partially Observable Markov Decision Processes.” In <em>52nd IEEE Conference on Decision and Control</em>, 2398–2405. <https://doi.org/10.1109/CDC.2013.6760239>.</li>
<li>Becker, Raphen, Shlomo Zilberstein, Victor Lesser, and Claudia V. Goldman. 2004. “Solving Transition Independent Decentralized Markov Decision Processes.” <em>Journal of Artificial Intelligence Research</em> 22: 423–55.</li>
<li>Bernstein, Daniel S., Robert Givan, Neil Immerman, and Shlomo Zilberstein. 2002. “The Complexity of Decentralized Control of Markov Decision Processes.” <em>Mathematics of Operations Research</em> 27 (4): 819–40.</li>
<li>Carlin, A., and S. Zilberstein. 2008. “Value-Based Observation Compression for DEC-POMDPs.” In <em>Proceedings of the Seventh International Conference on Autonomous Agents and Multiagent Systems</em>.</li>
<li>Chi, Cheng, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. 2024. “Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.” <em>The International Journal of Robotics Research</em>.</li>
<li>Dibangoye, J. S., A.-I. Mouaddib, and B. Chaib-draa. 2009. “Point-Based Incremental Pruning Heuristic for Solving Finite-Horizon DEC-POMDPs.” In <em>Proceedings of the Eighth International Conference on Autonomous Agents and Multiagent Systems</em>.</li>
<li>Hansen, Eric A, Daniel S Bernstein, and Shlomo Zilberstein. 2004. “Dynamic Programming for Partially Observable Stochastic Games.” In <em>AAAI</em>, 4:709–15.</li>
<li>Kumar, A., and S. Zilberstein. 2010. “Point-Based Backup for Decentralized POMDPs: Complexity and New Algorithms.” In <em>Proceedings of the Ninth International Conference on Autonomous Agents and Multiagent Systems</em>, 1315–22.</li>
<li>Madani, Omid, Steve Hanks, and Anne Condon. 1999. “On the Undecidability of Probabilistic Planning and Infinite-Horizon Partially Observable Markov Decision Problems.” In, 541–48. AAAI ’99/IAAI ’99. Orlando, Florida, USA: American Association for Artificial Intelligence.</li>
<li>Nair, Ranjit, Milind Tambe, Makoto Yokoo, David V. Pynadath, and Stacy Marsella. 2005. “Networked Distributed POMDPs: A Synthesis of Distributed Constraint Optimization and POMDPs.” In <em>Proceedings of the 20th National Conference on Artificial Intelligence (AAAI-05)</em>, 133–39. AAAI Press.</li>
<li>Nair, Ranjit, Milind Tambe, Makoto Yokoo, David Pynadath, and Stacy Marsella. 2003. “Taming Decentralized POMDPs: Towards Efficient Policy Computation for Multiagent Settings.” In <em>IJCAI</em>, 3:705–11.</li>
<li>Papadimitriou, Christos H., and John N. Tsitsiklis. 1987. “The Complexity of Markov Decision Processes.” <em>Mathematics of Operations Research</em> 12 (3): 441–50. <https://doi.org/10.1287/moor.12.3.441>.</li>
<li>Seuken, Sven, and Shlomo Zilberstein. 2007a. “Improved Memory-Bounded Dynamic Programming for Decentralized POMDPs,” 344–51.</li>
<li> 2007b. “Memory-Bounded Dynamic Programming for DECPOMDPs.” <em>Proceedings of the 20th International Joint Conference on Artificial Intelligence (IJCAI)</em>, 2009–15.</li>
<li>Sutton, Richard S., and Andrew G. Barto. 2018. <em>Reinforcement Learning: An Introduction</em>. 2nd ed. MIT Press.</li>
<li>Szer, Daniel, François Charpillet, and Shlomo Zilberstein. 2005. “MAA*: A Heuristic Search Algorithm for Solving Decentralized POMDPs.” In <em>Proceedings of the 21st Conference on Uncertainty in Artificial Intelligence (UAI)</em>, 576–90. AUAI Press.</li>
<li>Wu, F., S. Zilberstein, and X. Chen. 2010. “Point-Based Policy Generation for Decentralized POMDPs.” In <em>Proceedings of the Ninth International Conference on Autonomous Agents and Multiagent Systems</em>, 1307–14.</li>
{{< /references >}}



</div>
<!-- moved to root content -->
<!-- moved back under rl/marl/ -->
