---
title: "Optimal Q Value Functions for Dec POMDP"
math: true
---

{{< katex />}}

# Notions

|                                             |                                                                                                            |
|:--------------------------------------------|:-----------------------------------------------------------------------------------------------------------|
| $s^t$                                       | the state at $t$ with problem horizon $h$                                                                  |
| $o^t$                                       | the joint observation of agents $o^t=\langle o_1^t, \dots, o_n^t \rangle$ at $t$                           |
| $\mathcal{O}$                               | the joint observation space                                                                                |
| $\vec{\theta}^t$                            | the joint observation-action history until $t$, $\vec{\theta}^t=(o^0, a^0, \dots, o^t)$                    |
| $\vec{\Theta}^t$                            | the joint history space at $t$                                                                             |
| $\vec{\Theta}^t_\pi$                        | the set of $\vec{\theta}^t$ consistent with policy $\pi$                                                   |
|                                             |                                                                                                            |
| $\delta^{t}$                                | the decision rule (a temporal structure of policy) at $t$                                                  |
| $\delta^{t,*}$                              | the optimal decision rule at $t$ following $\psi^{t-1, *}$                                                 |
| $\delta^{t,\circledast}_\psi$               | the optimal decision rule at $t$ following $\psi^{t-1}$                                                    |
| $\Delta^t$                                  | the decision rule space at $t$                                                                             |
| $\psi^{t}$                                  | the past joint policy until $t$, $\psi^{t} = \delta^{[0, t)}$                                              |
| $\psi^{t, *}$                               | the optimal past joint policy until $t$, $\psi^{t, *} = \delta^{[0, t), *}$                                |
| $\psi^{t, \circledast}$                     | the past joint policy until $t$ with non-optimal $\psi^{t-1}$ and optimal $\delta^{t-1, \circledast}_\psi$ |
| $\Psi^{t}$                                  | the past joint policy space at $t$                                                                         |
| $\xi^t$                                     | the subsequent joint policy from $t$, $\xi^{t} = \delta^{[t, h)}$                                          |
| $\xi^{t, *}$                                | the optimal subsequent joint policy from $t$, $\xi^{t} = \delta^{[t, h), *}$                               |
| $\xi^{t, \circledast}_\psi$                 | the optimal subsequent joint policy from $t$ following non-optimal $\psi^t$                                |
| $\pi$                                       | the joint pure policy $\pi=\delta^{[0, h)}$                                                                |
| $\pi^*$                                     | the joint optimal pure policy $\pi^*=\delta^{[0, h), *}$                                                   |
|                                             |                                                                                                            |
| $R(\vec{\theta}^t, \psi^{t+1})$             | the immediate reward function following $\psi^{t+1}$                                                       |
| $Q(\vec{\theta}^t, \psi^{t+1})$             | the history-policy value function following $\psi^{t+1}$                                                   |
| $Q^*(\vec{\theta}^t, \psi^{t+1})$           | the optimal history-policy value function following $\psi^{t+1}$                                           |
| $Q^\circledast(\vec{\theta}^t, \psi^{t+1})$ | the sequentially rational optimal history-policy value function following $\psi^{t+1}$                     |

# Normative Optimal Q-Value Function

<div id="defn:normative-Q" class="definition">

**Definition 1**. *The optimal Q-value function $Q^*$ in Dec-POMDP, the expected cumulative reward over time steps $[t,h)$ induced by optimal joint policy $\pi^{*}$, $\forall \vec{\theta}^t\in \vec{\Theta}^t_{\psi^{t, *}}, \forall \psi^{t+1}\in(\psi^{t, *},\Delta^t)$, is defined as, $$Q^*(\vec{\theta}^t, \psi^{t+1}) = \left\{
        \begin{aligned}
        &R(\vec{\theta}^t, \psi^{t+1}), &t=h-1 \\ 
        &R(\vec{\theta}^t, \psi^{t+1}) + \sum_{o^{t+1} \in \mathcal{O}} P(o^{t+1}|\vec{\theta}^t, \psi^{t+1}) Q^*(\vec{\theta}^{t+1}, \pi^*(\vec{\theta}^{t+1})). &0\leqslant t < h-1 \\
        \end{aligned}
        \right .\label{eq:normative-Q}$$*

</div>

Here, $\pi^*(\vec{\theta}^{t+1})\equiv \psi^{t+2, *}$ because of the consistent optimality of policy.

<div id="prop:problem" class="proposition">

**Proposition 1**. *In Dec-POMDP, deriving an optimal policy from the normative optimal history-policy value function defined in Equ. <a href="#eq:normative-Q" data-reference-type="ref" data-reference="eq:normative-Q">[eq:normative-Q]</a> is impractical (clarifying Sec. 4.3.3, (Oliehoek, Spaan, and Vlassis 2008)).*

</div>

<div class="proof">

*Proof.* We check the optima in 2 steps. The independent and dependent variables are marked in red.

To calculate the Pareto optima of Bayesian game at $t$, $$\textcolor{red}{\delta^{t, *}}
    = \mathop{\mathrm{argmax}}_{\delta^t}\sum_{\vec{\theta}^t \in \vec{\Theta}^t_{\psi^{t, *}}} P(\vec{\theta}^t|\psi^{t, *}) \textcolor{red}{Q^*}(\vec{\theta}^t, (\psi^{t, *}, \delta^t)),$$ note that calculating $\delta^{t,*}$ depends on $\psi^{t, *} = \delta^{[0, t), *}$ and $Q^*(\vec{\theta}^t, \cdot)$.

According to Definition. <a href="#defn:normative-Q" data-reference-type="ref" data-reference="defn:normative-Q">1</a>, the optimal Bellman equation can be written as, $$\textcolor{red}{Q^*}(\vec{\theta}^t, \psi^{t+1}) = R(\vec{\theta}^t, \psi^{t+1}) + \sum_{o^{t+1} \in \mathcal{O}} P(o^{t+1}|\vec{\theta}^t, \psi^{t+1}) \max_{\delta^{t+1}}Q^*(\vec{\theta}^{t+1}, (\textcolor{red}{\psi^{t+1, *}}, \delta^{t+1})),$$ when $0\leqslant t < h-1$. This indicates that $Q^*(\vec{\theta}^t, \cdot)$ depends on $\psi^{t+1, *}$. Consequently, calculating $\delta^{t,*}$ inherently depends on $\delta^{[0, t], *}$ (includes itself), making it self-dependent and impractical to solve. ◻

{{< hint info >}}
Note: The dependency of $P(o^{t+1}\mid\vec{\theta}^t, \psi^{t+1})$ is not problematic and can be handled analogously to how the stochasticity $P(s^{t+1}\mid s^t, a)$ is treated via double learning (Sutton and Barto 2018, Sec. 6.7).
{{< /hint >}}

{{< hint info >}}
Single-agent (PO)MDP, where belief states are available, does not have this issue because the Q-value need not be history-dependent (Markov property).
{{< /hint >}}

</div>

# Sequentially Rational Optimal Q-Value Function

To make optimal Q-value in Dec-POMDP computable, (Oliehoek, Spaan, and Vlassis 2008) defined another form of Q-value function and eliminated the dependency on past optimality.

<div class="definition">

**Definition 2**. *The sequentially rational optimal Q-value function $Q^\circledast$ in Dec-POMDP, the expected cumulative reward over time steps $[t,h)$ induced by optimal subsequent joint policy $\xi^{t, \circledast}_\psi$, $\forall \vec{\theta}^t\in \vec{\Theta}^t_{\Psi^{t}}, \forall\psi^{t+1}\in\Psi^{t+1}$, is defined as, $$Q^\circledast(\vec{\theta}^t, \psi^{t+1}) = \left\{
        \begin{aligned}
        &R(\vec{\theta}^t, \psi^{t+1}), &t=h-1\\ 
        &R(\vec{\theta}^t, \psi^{t+1}) + \sum_{o^{t+1} \in \mathcal{O}} P(o^{t+1}|\vec{\theta}^t, \psi^{t+1}) Q^\circledast(\vec{\theta}^{t+1}, \psi^{t+2, \circledast}), &0\leqslant t < h-1 \\
        \end{aligned}
        \right .\label{eq:SR-Q}$$ where $\psi^{t+2, \circledast}=(\psi^{t+1}, \delta^{t+1, \circledast}_{\psi}), \forall \ \psi^{t+1} \in \Psi^{t+1}$.*

</div>

Note that the only difference of $Q^\circledast$ from $Q^*$ is $\psi^{t+2, \circledast}$, consequently expanding $Q^*$’s candidates of history from $\vec{\theta}^t \in \vec{\Theta}^t_{\psi^{t, *}}$ to $\vec{\theta}^t \in \vec{\Theta}^t_{\Psi^{t}}$ and policy from $\psi^{t+1}\in(\psi^{t, *},\Delta^t)$ to $\psi^{t+1}\in(\Psi^t,\Delta^t)$.

Beyond solving the problem of Proposition <a href="#prop:problem" data-reference-type="ref" data-reference="prop:problem">1</a>, another advantage of $Q^\circledast$ is that it allows for the computation of optimal subsequent policy $\xi^{t, *}_\psi$ following any past policy $\psi^{t}$. This is beneficial in online applications where agents may occasionally deviate from the optimal policy.

# Open Questions

- We have seen some advantages of defining the optimal Q-value function as $Q^\circledast$, what are the downsides to defining it this way (e.g., high computational costs)?

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-Oliehoek08JAIR" class="csl-entry">

Oliehoek, Frans A., Matthijs T. J. Spaan, and Nikos Vlassis. 2008. “Optimal and Approximate Q-Value Functions for Decentralized POMDPs.” *Journal of Artificial Intelligence Research* 32: 289–353.

</div>

<div id="ref-sutton2018reinforcement" class="csl-entry">

Sutton, Richard S., and Andrew G. Barto. 2018. *Reinforcement Learning: An Introduction*. 2nd ed. MIT Press.

</div>

</div>

<!-- footnotes converted to hints above -->
<!-- migrated from leaf-bundle to single-file naming -->
