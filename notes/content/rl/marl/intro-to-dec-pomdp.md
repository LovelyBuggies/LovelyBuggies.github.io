---
date: 2024-12-21
title: "Intro to Dec-POMDP"
math: true
postType: review
linkTitle: "Intro to Dec-POMDP"
readingTime: 10
weight: 2
---

{{< katex />}}

# An Introduction to Dec-POMDP
{{< postbadges >}}

## Dec-POMDP

<div class="definition">
  <strong>Definition 1</strong>. A Dec-POMDP is a tuple {{< katex >}}\langle \mathbb{I}, \mathcal{S}, \{\mathbb{A}_i\}, T, R, \{\mathbb{O}_i\}, O, \mathcal{H}, \gamma\rangle{{< /katex >}}:

- {{< katex >}}\mathbb{I}{{< /katex >}} is a finite sets of agents, {{< katex >}}|\mathbb{I}|=n{{< /katex >}};

- {{< katex >}}\mathcal{S}{{< /katex >}} is a set of states with designated initial state distribution {{< katex >}}b^0{{< /katex >}};

- {{< katex >}}\mathbb{A}_i{{< /katex >}} is a set of actions for agent {{< katex >}}i{{< /katex >}} with {{< katex >}}\mathbb{A}\doteq \times_i \mathbb{A}_i{{< /katex >}} the set of joint actions;

- {{< katex >}}T{{< /katex >}} is the state transition probability function, {{< katex >}}T{{< /katex >}}: {{< katex >}}\mathcal{S} \times \mathbb{A} \times \mathcal{S} \rightarrow [0, 1]{{< /katex >}}, that specifies the probability of transitioning from state {{< katex >}}s \in \mathcal{S}{{< /katex >}} to {{< katex >}}s' \in \mathcal{S}{{< /katex >}} when the actions {{< katex >}}\boldsymbol{a} \in \mathbb{A}{{< /katex >}} are taken by agents (i.e., {{< katex >}}T(s, \boldsymbol{a}, s')=P(s'|s, \textbf{a}){{< /katex >}});

- {{< katex >}}R{{< /katex >}} is the joint reward function, where {{< katex >}}R{{< /katex >}}: {{< katex >}}\mathcal{S} \times \mathbb{A} \rightarrow \mathbb{R}{{< /katex >}};

- {{< katex >}}\mathbb{O}_i{{< /katex >}} is a set of observations for each agent {{< katex >}}i{{< /katex >}}, with {{< katex >}}\mathbb{O}\doteq\times_i \mathbb{O}_i{{< /katex >}} the set of joint observations;

- {{< katex >}}O{{< /katex >}} is an observation probability function, {{< katex >}}O{{< /katex >}}: {{< katex >}}\mathbb{A} \times \mathcal{S} \times \mathbb{O}{{< /katex >}}, that specifies the probability of seeing observation {{< katex >}}\boldsymbol{o}' \in \mathbb{O}{{< /katex >}} given the actions {{< katex >}}\boldsymbol{a} \in \mathbb{A}{{< /katex >}} are taken and state {{< katex >}}s' \in \mathcal{S}{{< /katex >}} is observed (i.e., {{< katex >}}O(\boldsymbol{a}, s', \boldsymbol{o}')=P(\boldsymbol{o}'|\textbf{a},s'){{< /katex >}});

- {{< katex >}}\mathcal{H}{{< /katex >}} is the horizon (the number of steps until termination);

- {{< katex >}}\gamma{{< /katex >}} is the discount factor for the return.

In Dec-POMDP, since the states are not directly observable, each agent maintains its local observation-action history, {{< katex >}}h_i^t=\{a_i^{0}, o_i^{1}, \cdots o_i^t\} {{< /katex >}}, to infer information about the state. A solution to a Dec-POMDP is a joint policy $\boldsymbol{\pi}:\mathbb{H}_i\to\mathbb{A}_i, \forall i \in \mathbb{I}$ over joint observation-action history {{< katex >}}\boldsymbol{h}=\{\boldsymbol{a}^{0}, \boldsymbol{o}^{1}, \cdots \boldsymbol{o}^{\mathcal{H}-1}\}{{< /katex >}}, where an optimal solution maximizes the expected return, 

{{< katex display=true >}}
\boldsymbol{\pi}^*=\mathop{\mathrm{argmax}}_{\boldsymbol{\pi}}\mathbb{E}\left[\textstyle\sum_{t=0}^{\mathcal{H}-1}R(\boldsymbol{h}, \boldsymbol{\pi}(\boldsymbol{h}))\middle|b^0\right].
{{< /katex >}}

</div>

In Dec-POMDP, the Bellman recursive formulation of the history V-function is,

{{< katex display=true >}}
\label{eq:decpomdp-V}
\begin{aligned}
    V^{\boldsymbol{\pi}}(\boldsymbol{h}) &= \sum_{s} P(s|b^0, \boldsymbol{h})\left[R(s, \boldsymbol{\pi}(\boldsymbol{h}))+\gamma \sum_{s'}P(s'|s, \boldsymbol{\pi}(\boldsymbol{h})) \sum_{\boldsymbol{o}'}P(\boldsymbol{o}'|\boldsymbol{\pi}(\boldsymbol{h}), s') V^{\boldsymbol{\pi}}(\boldsymbol{h}') \right]\\
    &\equiv R(\boldsymbol{h}, \boldsymbol{\pi})+\gamma\sum_{\boldsymbol{o}'}P(\boldsymbol{o}'|\boldsymbol{h}, \boldsymbol{\pi}) V^{\boldsymbol{\pi}}(\boldsymbol{h}').
\end{aligned}
{{< /katex >}}

The definition of the value function is flexible: it can be based on the value of a state, a belief state, an observation, a state history, an observation history, a single action (single-step policy), a full policy (action history), observation–action history, or combinations.

the Bellman recursive formulation of the **history-policy** Q-function is,

{{< katex display=true >}}
\label{eq:decpomdp-q}
\small
\begin{aligned}
    Q^{\boldsymbol{\pi}}(\boldsymbol{h}, \boldsymbol{\pi}) &= \sum_{s} P(s|b^0, \boldsymbol{h})\left[R(s, \boldsymbol{\pi}(\boldsymbol{h}))+\gamma\sum_{s'}P(s'|s, \boldsymbol{\pi}(\boldsymbol{h})) \sum_{\boldsymbol{o}'}P(\boldsymbol{o}'|\boldsymbol{\pi}(\boldsymbol{h}), s') Q^{\boldsymbol{\pi}}(\boldsymbol{h}', \boldsymbol{\pi}(\boldsymbol{h}')) \right]\\
    &\equiv R(\boldsymbol{h}, \boldsymbol{\pi})+\gamma\sum_{\boldsymbol{o}'}P(\boldsymbol{o}'|\boldsymbol{h}, \boldsymbol{\pi}) Q^{\boldsymbol{\pi}}(\boldsymbol{h}', \boldsymbol{\pi}) .
\end{aligned}
{{< /katex >}}

{{< sidenote id="curse-of-dimension" >}}

<strong> Fun fact:</strong> You might find a paragraph that’s over 66.67% similar to this one in most of <a href="https://www.ccs.neu.edu/home/camato/publications.html">our group’s papers</a> :3

{{< /sidenote >}}

## Special Classes of Dec-POMDP 

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


{{% details title="Related Theorems" open=false %}}

<span id="tab:complexity" label="tab:complexity"></span>

<div class="theorem">
<strong>Theorem 1</strong>. An MDP is P-complete in finite and infinite horizons (Papadimitriou and Tsitsiklis 1987).

<strong>Theorem 2</strong>. A finite POMDP is PSPACE-complete (Papadimitriou and Tsitsiklis 1987).

<strong>Theorem 3</strong>. The complexity of an infinite POMDP is undecidable (Madani, Hanks, and Condon 1999), leading to the undecidability of the infinite Dec-POMDP complexity.

<strong>Theorem 4</strong>. A finite Dec-POMDP ($n\geqslant2$) is NEXP-complete, and a finite Dec-MDP ($n\geqslant3$) is also NEXP-complete (Bernstein et al. 2002).

<strong>Fact 1</strong>. A Dec-MDP with TI and RI can be solved independently, resulting in P-complete.

<strong>Theorem 5</strong>. A Dec-MDP with TI and joint reward is NP-complete, a Dec-MDP with RI but no TI is NEXP-complete (Becker et al. 2004).

<strong>Fact 2</strong>. An ND-POMDP has the same worst-case complexity as a Dec-POMDP (Nair et al. 2005).

</div>

{{% /details %}}


## References

{{< references >}}
<li>Amato, C., Chowdhary, G., Geramifard, A., Üre, N. K., & Kochenderfer, M. J. (2013, December). Decentralized control of partially observable Markov decision processes. In 52nd IEEE Conference on Decision and Control (pp. 2398-2405). IEEE.</li>
<li>Becker, R., Zilberstein, S., Lesser, V., & Goldman, C. V. (2004). Solving transition independent decentralized Markov decision processes. Journal of Artificial Intelligence Research, 22, 423-455.</li>
<li>Bernstein, D. S., Givan, R., Immerman, N., & Zilberstein, S. (2002). The complexity of decentralized control of Markov decision processes. Mathematics of operations research, 27(4), 819-840.</li>
<li>Carlin, A., & Zilberstein, S. (2008, May). Value-based observation compression for DEC-POMDPs. In Proceedings of the 7th international joint conference on Autonomous agents and multiagent systems-Volume 1 (pp. 501-508).</li>
<li>Chi, C., Xu, Z., Feng, S., Cousineau, E., Du, Y., Burchfiel, B., ... & Song, S. (2023). Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, 02783649241273668.</li>
<li>Dibangoye, J. S., Mouaddib, A. I., & Chai-draa, B. (2009, May). Point-based incremental pruning heuristic for solving finite-horizon DEC-POMDPs. In Proceedings of The 8th International Conference on Autonomous Agents and Multiagent Systems-Volume 1 (pp. 569-576).</li>
<li>Hansen, E. A., Bernstein, D. S., & Zilberstein, S. (2004, July). Dynamic programming for partially observable stochastic games. In AAAI (Vol. 4, pp. 709-715).</li>
<li>Kumar, A., & Zilberstein, S. (2010). Point-based backup for decentralized POMPDs: Complexity and new algorithms.</li>
<li>Madani, O., Hanks, S., & Condon, A. (1999). On the undecidability of probabilistic planning and infinite-horizon partially observable Markov decision problems. Aaai/iaai, 10(315149.315395).</li>
<li>Nair, R., Varakantham, P., Tambe, M., & Yokoo, M. (2005, July). Networked distributed POMDPs: A synthesis of distributed constraint optimization and POMDPs. In AAAI (Vol. 5, pp. 133-139).</li>
<li>Nair, R., Tambe, M., Yokoo, M., Pynadath, D., & Marsella, S. (2003, August). Taming decentralized POMDPs: Towards efficient policy computation for multiagent settings. In IJCAI (Vol. 3, pp. 705-711).</li>
<li>Papadimitriou, C. H., & Tsitsiklis, J. N. (1987). The complexity of Markov decision processes. Mathematics of operations research, 12(3), 441-450.</li>
<li>Seuken, S., & Zilberstein, S. (2012). Improved memory-bounded dynamic programming for decentralized POMDPs. arXiv preprint arXiv:1206.5295.</li>
<li>Seuken, S., & Zilberstein, S. (2007). Memory-Bounded Dynamic Programming for DEC-POMDPs. In IJCAI (pp. 2009-2015).</li>
<li>Sutton, R. S., & Barto, A. G. (1998). Reinforcement learning: An introduction (Vol. 1, No. 1, pp. 9-11). Cambridge: MIT press.</li>
<li>Szer, D., Charpillet, F., & Zilberstein, S. (2012). MAA*: A heuristic search algorithm for solving decentralized POMDPs. arXiv preprint arXiv:1207.1359.</li>
<li>Wu, F., Zilberstein, S., & Chen, X. (2010, May). Point-based policy generation for decentralized POMDPs. In Proceedings of the 9th International Conference on Autonomous Agents and Multiagent Systems: volume 1-Volume 1 (pp. 1307-1314).</li>
{{< /references >}}



</div>
<!-- moved to root content -->
<!-- moved back under rl/marl/ -->
