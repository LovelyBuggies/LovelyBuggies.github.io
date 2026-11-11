---
date: 2024-12-21
title: "An Introduction to Dec-POMDP"
description: "An introduction to decentralized partially observable Markov decision processes, their computational complexity, and representations of joint policies."
math: true
postType: review
readingTime: 10
---

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
    V^{\boldsymbol{\pi}}(\boldsymbol{h}) &= \sum_{s} P(s|b^0, \boldsymbol{h})\Bigl[R(s, \boldsymbol{\pi}(\boldsymbol{h}))\\
    &\quad {}+\gamma \sum_{s'}P(s'|s, \boldsymbol{\pi}(\boldsymbol{h}))\\
    &\qquad \sum_{\boldsymbol{o}'}P(\boldsymbol{o}'|\boldsymbol{\pi}(\boldsymbol{h}), s') V^{\boldsymbol{\pi}}(\boldsymbol{h}') \Bigr]\\
    &\equiv R(\boldsymbol{h}, \boldsymbol{\pi})+\gamma\sum_{\boldsymbol{o}'}P(\boldsymbol{o}'|\boldsymbol{h}, \boldsymbol{\pi}) V^{\boldsymbol{\pi}}(\boldsymbol{h}').
\end{aligned}
{{< /katex >}}

The definition of the value function is flexible: it can be based on the value of a state, a belief state, an observation, a state history, an observation history, a single action (single-step policy), a full policy (action history), observation–action history, or combinations.

the Bellman recursive formulation of the **history-policy** Q-function is,

{{< katex display=true >}}
\label{eq:decpomdp-q}
\small
\begin{aligned}
    Q^{\boldsymbol{\pi}}(\boldsymbol{h}, \boldsymbol{\pi}) &= \sum_{s} P(s|b^0, \boldsymbol{h})\Bigl[R(s, \boldsymbol{\pi}(\boldsymbol{h}))\\
    &\quad {}+\gamma\sum_{s'}P(s'|s, \boldsymbol{\pi}(\boldsymbol{h}))\\
    &\qquad \sum_{\boldsymbol{o}'}P(\boldsymbol{o}'|\boldsymbol{\pi}(\boldsymbol{h}), s') Q^{\boldsymbol{\pi}}(\boldsymbol{h}', \boldsymbol{\pi}(\boldsymbol{h}')) \Bigr]\\
    &\equiv R(\boldsymbol{h}, \boldsymbol{\pi})+\gamma\sum_{\boldsymbol{o}'}P(\boldsymbol{o}'|\boldsymbol{h}, \boldsymbol{\pi}) Q^{\boldsymbol{\pi}}(\boldsymbol{h}', \boldsymbol{\pi}) .
\end{aligned}
{{< /katex >}}

{{< sidenote id="curse-of-dimension" >}}

<strong> Fun fact:</strong> You might find a paragraph that’s over 66.67% similar to this one in most of <a href="https://www.ccs.neu.edu/home/camato/publications.html">our group’s papers</a> :3

{{< /sidenote >}}

## Special Classes of Dec-POMDP

- **Centralized:** MMDP is a fully observable version of Dec-POMDP, but it does not specify decentralized control. Dec-MDP assumes that the joint observations uniquely determine the state, while agents still act with local observations. Similarly, MPOMDP does not specify whether the control is decentralized, which could have a centralized policy {{< katex >}}\mathbb{H}\to\mathbb{A}{{< /katex >}}.
- **Decentralized:** A decentralized control model might be factorized with independent local variables, e.g., transition-independence (TI) {{< katex >}}T(s, \boldsymbol{a}, s')=\Pi_{i=1}^{n} T(s_i, a_i, s_i'){{< /katex >}} and reward-independence (RI) {{< katex >}}R(s,\boldsymbol{\pi})=f_\text{mono}(\langle R(s_i, \pi_i)\rangle_{i=1}^{n}){{< /katex >}}. The networked distributed POMDP ([Nair et al., 2005](https://cdn.aaai.org/AAAI/2005/AAAI05-022.pdf)) represents the factored one with TI and block-RI, i.e., {{< katex >}}R(s,\boldsymbol{\pi})=f_\text{mono}(\langle R(s_{i, \mathcal{N}(i)}, \pi_{i, \mathcal{N}(i)})\rangle_{i=1}^{n}){{< /katex >}}, where {{< katex >}}{\mathcal{N}(i)}{{< /katex >}} are the neighbors of {{< katex >}}i{{< /katex >}}.

[Amato et al. (2013)](https://people.csail.mit.edu/camato/publications/DecPOMDP-survey-final.pdf) summarize the **worst-case complexity** of finite-horizon problems. The TI Dec-MDP rows below assume independent observations as well as independent transitions.

<div id="tab:complexity">

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

</div>

{{% details title="Related Theorems" open=false %}}

<div class="theorem">

<strong>Theorem 1</strong>. An MDP is P-complete in finite and infinite horizons (<a href="https://doi.org/10.1287/moor.12.3.441">Papadimitriou and Tsitsiklis 1987</a>).

<strong>Theorem 2</strong>. A finite POMDP is PSPACE-complete (<a href="https://doi.org/10.1287/moor.12.3.441">Papadimitriou and Tsitsiklis 1987</a>).

<strong>Theorem 3</strong>. The complexity of an infinite POMDP is undecidable (<a href="https://cdn.aaai.org/AAAI/1999/AAAI99-077.pdf">Madani, Hanks, and Condon 1999</a>), leading to the undecidability of the infinite Dec-POMDP complexity.

<strong>Theorem 4</strong>. A finite Dec-POMDP ($n\geqslant2$) is NEXP-complete, and a finite Dec-MDP ($n\geqslant3$) is also NEXP-complete (<a href="https://doi.org/10.1287/moor.27.4.819.297">Bernstein et al. 2002</a>).

<strong>Fact 1</strong>. With independent transitions, observations, and rewards, a Dec-MDP can be solved independently for each agent and is P-complete (<a href="https://people.csail.mit.edu/camato/publications/DecPOMDP-survey-final.pdf">Amato et al. 2013, Theorem 5</a>).

<strong>Theorem 5</strong>. A Dec-MDP with independent transitions and observations but joint reward is NP-complete (<a href="https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume22/becker04a.pdf">Becker et al. 2004</a>). Reward independence alone leaves the problem NEXP-complete (<a href="https://proceedings.neurips.cc/paper/2009/hash/fec8d47d412bcbeece3d9128ae855a7a-Abstract.html">Allen and Zilberstein 2009</a>).

<strong>Fact 2</strong>. An ND-POMDP has the same worst-case complexity as a Dec-POMDP (<a href="https://people.csail.mit.edu/camato/publications/DecPOMDP-survey-final.pdf">Amato et al. 2013, Table I</a>).

</div>

{{% /details %}}
<!-- moved to root content -->
<!-- moved back under rl/marl/ -->
