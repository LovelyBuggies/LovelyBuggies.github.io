---
date: 2024-11-11
title: "Optimal Q Value Functions for Dec-POMDP"
math: true
postType: thoughts
linkTitle: "Optimal Q Value Functions for Dec-POMDP"
readingTime: 4
---

{{< katex />}}

# Optimal Q Value Functions for Dec-POMDP
{{< postbadges >}}

## Notions

{{< tabs >}}

{{% tab "States & Observations" %}}
- {{% katex %}}s^t{{% /katex %}}: state at time {{% katex %}}t{{% /katex %}} with horizon {{% katex %}}h{{% /katex %}}.
- {{% katex %}}o^t{{% /katex %}}: joint observation {{% katex %}}o^t = \langle o_1^t, \dots, o_n^t \rangle{{% /katex %}} at {{% katex %}}t{{% /katex %}}.
- {{% katex %}}\mathcal{O}{{% /katex %}}: joint observation space.
- {{% katex %}}\vec{\theta}^t{{% /katex %}}: joint observation–action history until {{% katex %}}t{{% /katex %}}, {{% katex %}}\vec{\theta}^t=(o^0, a^0, \dots, o^t){{% /katex %}}.
- {{% katex %}}\vec{\Theta}^t{{% /katex %}}: joint history space at {{% katex %}}t{{% /katex %}}.
- {{% katex %}}\vec{\Theta}^t_\pi{{% /katex %}}: set of {{% katex %}}\vec{\theta}^t{{% /katex %}} consistent with policy {{% katex %}}\pi{{% /katex %}}.
{{% /tab %}}

{{% tab "Policy & History" %}}
- {{% katex %}}\delta^{t}{{% /katex %}}: decision rule (temporal policy component) at {{% katex %}}t{{% /katex %}}.
- {{% katex %}}\delta^{t,*}{{% /katex %}}: optimal decision rule at {{% katex %}}t{{% /katex %}} given {{% katex %}}\psi^{t-1,*}{{% /katex %}}.
- {{% katex %}}\delta^{t,\circledast}_\psi{{% /katex %}}: optimal decision at {{% katex %}}t{{% /katex %}} given non‑optimal {{% katex %}}\psi^{t-1}{{% /katex %}}.
- {{% katex %}}\Delta^t{{% /katex %}}: decision rule space at {{% katex %}}t{{% /katex %}}.
- {{% katex %}}\psi^{t} = \delta^{[0,t)}{{% /katex %}}: past joint policy until {{% katex %}}t{{% /katex %}}.
- {{% katex %}}\psi^{t,*} = \delta^{[0,t),*}{{% /katex %}}: optimal past joint policy until {{% katex %}}t{{% /katex %}}.
- {{% katex %}}\psi^{t,\circledast}{{% /katex %}}: past joint policy with non‑optimal {{% katex %}}\psi^{t-1}{{% /katex %}} and optimal {{% katex %}}\delta^{t-1,\circledast}_\psi{{% /katex %}}.
- {{% katex %}}\Psi^{t}{{% /katex %}}: past joint policy space at {{% katex %}}t{{% /katex %}}.
- {{% katex %}}\xi^{t} = \delta^{[t,h)}{{% /katex %}}: subsequent joint policy from {{% katex %}}t{{% /katex %}}.
- {{% katex %}}\xi^{t,*} = \delta^{[t,h),*}{{% /katex %}}: optimal subsequent joint policy from {{% katex %}}t{{% /katex %}}.
- {{% katex %}}\xi^{t,\circledast}_\psi{{% /katex %}}: optimal subsequent policy from {{% katex %}}t{{% /katex %}} given non‑optimal {{% katex %}}\psi^t{{% /katex %}}.
- {{% katex %}}\pi = \delta^{[0,h)}{{% /katex %}}: joint pure policy.
- {{% katex %}}\pi^* = \delta^{[0,h),*}{{% /katex %}}: joint optimal pure policy.
{{% /tab %}}

{{% tab "Rewards & Q" %}}
- {{% katex %}}R(\vec{\theta}^t, \psi^{t+1}){{% /katex %}}: immediate reward under {{% katex %}}\psi^{t+1}{{% /katex %}}.
- {{% katex %}}Q(\vec{\theta}^t, \psi^{t+1}){{% /katex %}}: history–policy value under {{% katex %}}\psi^{t+1}{{% /katex %}}.
- {{% katex %}}Q^*(\vec{\theta}^t, \psi^{t+1}){{% /katex %}}: optimal history–policy value under {{% katex %}}\psi^{t+1}{{% /katex %}}.
- {{% katex %}}Q^{\circledast}(\vec{\theta}^t, \psi^{t+1}){{% /katex %}}: sequentially rational optimal history–policy value under {{% katex %}}\psi^{t+1}{{% /katex %}}.
{{% /tab %}}

{{< /tabs >}}

## Normative Optimal Q-Value Function

<div id="defn:normative-Q" class="definition">

**Definition 1**. *The optimal Q-value function $Q^*$ in Dec-POMDP, the expected cumulative reward over time steps $[t,h)$ induced by optimal joint policy $\pi^{*}$, $\forall \vec{\theta}^t\in \vec{\Theta}^t_{\psi^{t, *}}, \forall \psi^{t+1}\in(\psi^{t, *},\Delta^t)$, is defined as, {{< katex display=true >}}
Q^*(\vec{\theta}^t, \psi^{t+1}) = \left\{
        \begin{aligned}
        &R(\vec{\theta}^t, \psi^{t+1}), &t=h-1 \\ 
        &R(\vec{\theta}^t, \psi^{t+1}) + \sum_{o^{t+1} \in \mathcal{O}} P(o^{t+1}|\vec{\theta}^t, \psi^{t+1}) Q^*(\vec{\theta}^{t+1}, \pi^*(\vec{\theta}^{t+1})). &0\leqslant t < h-1 \\
        \end{aligned}
        \right .\label{eq:normative-Q}
{{< /katex >}}
*

</div>

Here, {{< katex >}}\pi^*(\vec{\theta}^{t+1})\equiv \psi^{t+2, *}{{< /katex >}} because of the consistent optimality of policy.

<div id="prop:problem" class="proposition">

**Proposition 1**. *In Dec-POMDP, deriving an optimal policy from the normative optimal history-policy value function defined in Equ. <a href="#eq:normative-Q" data-reference-type="ref" data-reference="eq:normative-Q">[eq:normative-Q]</a> is impractical (clarifying Sec. 4.3.3, (Oliehoek, Spaan, and Vlassis 2008)).*

</div>

<div class="proof">

*Proof.* We check the optima in 2 steps. The independent and dependent variables are marked in red.

To calculate the Pareto optima of Bayesian game at $t$, {{< katex display=true >}}
\textcolor{red}{\delta^{t, *}}
    = \mathop{\mathrm{argmax}}_{\delta^t}\sum_{\vec{\theta}^t \in \vec{\Theta}^t_{\psi^{t, *}}} P(\vec{\theta}^t|\psi^{t, *}) \textcolor{red}{Q^*}(\vec{\theta}^t, (\psi^{t, *}, \delta^t)),
{{< /katex >}}
 note that calculating {{< katex >}}\delta^{t,*}{{< /katex >}} depends on {{< katex >}}\psi^{t, *} = \delta^{[0, t), *}{{< /katex >}} and {{< katex >}}Q^*(\vec{\theta}^t, \cdot){{< /katex >}}.

According to Definition. <a href="#defn:normative-Q" data-reference-type="ref" data-reference="defn:normative-Q">1</a>, the optimal Bellman equation can be written as, {{< katex display=true >}}
\textcolor{red}{Q^*}(\vec{\theta}^t, \psi^{t+1}) = R(\vec{\theta}^t, \psi^{t+1}) + \sum_{o^{t+1} \in \mathcal{O}} P(o^{t+1}|\vec{\theta}^t, \psi^{t+1}) \max_{\delta^{t+1}}Q^*(\vec{\theta}^{t+1}, (\textcolor{red}{\psi^{t+1, *}}, \delta^{t+1})),
{{< /katex >}}
 when {{< katex >}}0\leqslant t < h-1{{< /katex >}}. This indicates that {{< katex >}}Q^*(\vec{\theta}^t, \cdot){{< /katex >}} depends on {{< katex >}}\psi^{t+1, *}{{< /katex >}}. Consequently, calculating {{< katex >}}\delta^{t,*}{{< /katex >}} inherently depends on {{< katex >}}\delta^{[0, t], *}{{< /katex >}} (includes itself), making it self-dependent and impractical to solve. ◻

{{< hint info >}}
Note: The dependency of {{< katex >}}P(o^{t+1}\mid\vec{\theta}^t, \psi^{t+1}){{< /katex >}} is not problematic and can be handled analogously to how the stochasticity {{< katex >}}P(s^{t+1}\mid s^t, a){{< /katex >}} is treated via double learning (Sutton and Barto 2018, Sec. 6.7).
{{< /hint >}}

{{< hint info >}}
Single-agent (PO)MDP, where belief states are available, does not have this issue because the Q-value need not be history-dependent (Markov property).
{{< /hint >}}

</div>

## Sequentially Rational Optimal Q-Value Function

To make optimal Q-value in Dec-POMDP computable, (Oliehoek, Spaan, and Vlassis 2008) defined another form of Q-value function and eliminated the dependency on past optimality.

<div class="definition">

**Definition 2**. *The sequentially rational optimal Q-value function $Q^\circledast$ in Dec-POMDP, the expected cumulative reward over time steps $[t,h)$ induced by optimal subsequent joint policy $\xi^{t, \circledast}_\psi$, $\forall \vec{\theta}^t\in \vec{\Theta}^t_{\Psi^{t}}, \forall\psi^{t+1}\in\Psi^{t+1}$, is defined as, {{< katex display=true >}}
Q^\circledast(\vec{\theta}^t, \psi^{t+1}) = \left\{
        \begin{aligned}
        &R(\vec{\theta}^t, \psi^{t+1}), &t=h-1\\ 
        &R(\vec{\theta}^t, \psi^{t+1}) + \sum_{o^{t+1} \in \mathcal{O}} P(o^{t+1}|\vec{\theta}^t, \psi^{t+1}) Q^\circledast(\vec{\theta}^{t+1}, \psi^{t+2, \circledast}), &0\leqslant t < h-1 \\
        \end{aligned}
        \right .\label{eq:SR-Q}
{{< /katex >}}
 where {{< katex >}}\psi^{t+2, \circledast}=(\psi^{t+1}, \delta^{t+1, \circledast}_{\psi}), \forall \ \psi^{t+1} \in \Psi^{t+1}{{< /katex >}}.*

</div>

Note that the only difference of {{< katex >}}Q^\circledast{{< /katex >}} from {{< katex >}}Q^*{{< /katex >}} is {{< katex >}}\psi^{t+2, \circledast}{{< /katex >}}, consequently expanding {{< katex >}}Q^*{{< /katex >}}’s candidates of history from {{< katex >}}\vec{\theta}^t \in \vec{\Theta}^t_{\psi^{t, *}}{{< /katex >}} to {{< katex >}}\vec{\theta}^t \in \vec{\Theta}^t_{\Psi^{t}}{{< /katex >}} and policy from {{< katex >}}\psi^{t+1}\in(\psi^{t, *},\Delta^t){{< /katex >}} to {{< katex >}}\psi^{t+1}\in(\Psi^t,\Delta^t){{< /katex >}}.

Beyond solving the problem of Proposition <a href="#prop:problem" data-reference-type="ref" data-reference="prop:problem">1</a>, another advantage of {{< katex >}}Q^\circledast{{< /katex >}} is that it allows for the computation of optimal subsequent policy {{< katex >}}\xi^{t, *}_\psi{{< /katex >}} following any past policy {{< katex >}}\psi^{t}{{< /katex >}}. This is beneficial in online applications where agents may occasionally deviate from the optimal policy.

## Open Questions

- We have seen some advantages of defining the optimal Q-value function as {{< katex >}}Q^\circledast{{< /katex >}}, what are the downsides to defining it this way (e.g., high computational costs)?



## References

{{< references >}}
<li>Oliehoek, Frans A., Matthijs T. J. Spaan, and Nikos Vlassis. 2008. “Optimal and Approximate Q-Value Functions for Decentralized POMDPs.” <em>Journal of Artificial Intelligence Research</em> 32: 289–353.</li>
<li>Sutton, Richard S., and Andrew G. Barto. 2018. <em>Reinforcement Learning: An Introduction</em>. 2nd ed. MIT Press.</li>
{{< /references >}}



<!-- footnotes converted to hints above -->
<!-- migrated from leaf-bundle to single-file naming -->
<!-- moved to root content -->
<!-- moved back under rl/marl/ -->
