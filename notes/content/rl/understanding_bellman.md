---
date: 2025-01-05
title: "Understanding Bellman"
math: true
weight: -100
postType: notes
linkTitle: "Understanding Bellman"
readingTime: 20
---

{{< katex />}}

# Understanding Bellman Equations
{{< postbadges >}}

## Bellman Equations

Bellman equations establish recusive relations between states and succeeding states, which can be applied as updating rules for value prediction.

<div class="definition">
  <strong>Definition.</strong> The Bellman equations for V-values are (Sutton and Barto 2018),



{{< katex display=true >}}

\begin{equation}

\begin{aligned}

V^\pi(s) &\doteq \mathbb{E}_{a \sim \pi(\cdot|s)} \left[ Q^\pi(s, a) \right] \\
&= \mathbb{E}_{a \sim \pi(\cdot|s)} \left[ R(s, a) + \gamma \mathbb{E}_{s' \sim P(\cdot|s,a)} \left[V^\pi(s')\right] \right] \\ 

V^*(s) &\doteq \max_{a} \left[ Q^*(s, a) \right] \\
&= \max_{a} \left[ R(s, a) + \gamma \mathbb{E}_{s' \sim P(\cdot|s,a)} \left[V^*(s')\right] \right].\nonumber

\end{aligned}

\end{equation}

{{< /katex >}}

The Bellman equations for Q-values are,

{{< katex display=true >}}

\begin{equation}

\begin{aligned}

Q^\pi(s, a) \doteq R(s, a) + \gamma \mathbb{E}_{s'\sim P(\cdot|s,a)} \left[V^\pi(s')\right] \\
= R(s, a) + \gamma \mathbb{E}_{s'\sim P(\cdot|s,a)} \left[\mathbb{E}_{a'\sim\pi(a'|s')} Q^\pi(s', a')\right] \\
Q^*(s, a) \doteq R(s, a) + \gamma \mathbb{E}_{s'\sim P(\cdot|s,a)} \left[V^*(s')\right] \\
= R(s, a) + \gamma \mathbb{E}_{s'\sim P(\cdot|s,a)} \left[\max_{a'} Q^*(s', a')\right].\nonumber

\end{aligned}

\end{equation}

{{< /katex >}}
</div>

$V^\pi(s)$ and $Q^\pi(s,a)$ are value representations following policy $\pi$, e.g., vectors and functions, and {{< katex display=true >}}
\tilde{\pi}(s) \doteq \mathop{\mathrm{argmax}}_a Q^\pi (s,a).

{{< /katex >}}

<div class="definition" id="curse-of-dimension">
<strong>The Curse of Dimension.</strong> Why do we mostly use MDP (where the future evolution is independent of its history) and hence Bellman Equations to model RL problems? (Bellman 1957) coined the “curse of dimension”, which describes the exponential increase in the state space size as dimensionality grows, making calculations extremely complex. Breaking this curse often requires altering the problem or its constraints, though complete solutions are not always achievable.
</div>


## Bellman Operators

A succinct representation is to define the Bellman Equation as a unary mathematical operator. The V-value Bellman and optimal Bellman Operators are,
{{< katex display=true >}}
(\mathcal{T}^\pi\circ V^\pi)(s) \doteq \mathbb{E}_{a \sim \pi(\cdot|s)} \left[ R(s, a) + \gamma \mathbb{E}_{s' \sim P(\cdot|s,a)} \left[V^\pi(s')\right] \right] \\
(\mathcal{T}^*\circ V^\pi)(s) \doteq \max_a \left[ R(s, a) + \gamma \mathbb{E}_{s' \sim P(\cdot|s,a)} \left[V^\pi(s')\right] \right]
{{< /katex >}}

The Bellman and optimal Bellman Operators {{< katex >}}\mathcal{T}^\pi{{< /katex >}} for Q-values are,

{{< katex display=true >}}
(\mathcal{T}^\pi\circ Q^\pi)(s, a) \doteq R(s, a) + \gamma \mathbb{E}_{s' \sim P(\cdot|s,a)} \left[ \mathbb{E}_{a' \sim \pi(a'|s')} Q^\pi(s', a') \right] \\
(\mathcal{T}^*\circ Q^\pi)(s, a) \doteq R(s, a) + \gamma \mathbb{E}_{s' \sim P(\cdot|s,a)} \left[ \max_{a'} Q^\pi(s', a') \right]
{{< /katex >}}

For convenience, we use Q-value as the representative in the following parts of this article.



## Bellman Properties

<div class="proposition">
<strong>Proposition 1</strong> ($\gamma$-contraction). Given any $Q,\ Q' \mapsto \mathbb{R}^{|\mathcal{S}| \times |\mathcal{A}|}$, Bellman Operators are $\gamma$-contraction Operators in $L^\infty$ norm, {{< katex display=true >}}
\begin{aligned}
        \|\mathcal{T}^\pi \circ Q - \mathcal{T}^\pi \circ Q'\|_\infty &\leqslant \gamma \|Q-Q'\|_\infty,\\
        \text{and }\|\mathcal{T}^* \circ Q - \mathcal{T}^* \circ Q'\|_\infty &\leqslant \gamma \|Q-Q'\|_\infty.
    \end{aligned}
{{< /katex >}}

<strong>Corollary 1</strong> (Fixed-point Iteration). For any {{< katex >}}Q^0 \mapsto \mathbb{R}^{|\mathcal{S}| \times |\mathcal{A}|}{{< /katex >}}, after {{< katex >}}k\to \infty{{< /katex >}} iterations of Bellman transformation, {{< katex >}}Q^{\pi, \infty} \doteq \lim_{k \to\infty} (\mathcal{T}^\pi)^k \circ Q^0{{< /katex >}}, or {{< katex >}}Q^{*, \infty} \doteq \lim_{k\to\infty} (\mathcal{T}^*)^k \circ Q^0{{< /katex >}}, according to <a href="https://en.wikipedia.org/wiki/Banach_fixed-point_theorem">Banach’s Fixed Point Theorem</a>:


{{< katex display=true >}}
Q^{\pi,\,\infty} = Q^{*,\,\infty} = Q^* ,
{{< /katex >}}

which uniquely satisfies,  {{< katex display=true >}}
\mathcal{T}^{\pi}(Q^*) = Q^*,  \text{or } \mathcal{T}^{*}(Q^*) = Q^* .{{< /katex >}}

<strong>Theorem 1</strong> (Fundamental theorem). Any memoryless policy that is greedy to {{< katex >}}Q^*{{< /katex >}} (<strong>deterministically</strong> maximizes) is optimal (Szepesvári 2010):


{{< katex display=true >}}
\tilde{\pi}^{*} \doteq \mathop{\mathrm{argmax}}_a Q^* = \pi^*.
{{< /katex >}}

**Proposition 2** (Monotone). *Bellman Operators are monotonic. For any Q-values {{< katex >}}Q,Q' \mapsto \mathbb{R}^{|\mathcal{S}| \times |\mathcal{A}|}{{< /katex >}}:*

{{< katex display=true >}}
\left (Q\leqslant Q'\right ) \Leftrightarrow \left (\mathcal{T}^\pi \circ Q\leqslant \mathcal{T}^\pi \circ Q'\right ) \\
\left (Q\leqslant Q'\right ) \Leftrightarrow \left (\mathcal{T}^* \circ Q\leqslant \mathcal{T}^* \circ Q'\right )
{{< /katex >}}

</div>

## Bellman Backup for Planning

### Dynamic Programming

According to the Fundamental Theorem, we can find {{< katex >}}\pi^*{{< /katex >}} efficiently once having access to {{< katex >}}Q^*{{< /katex >}}, without the need to find the policy whose Q-function **dominates** the others’ brute-force-ly. To avoid the Bellman Curse of Dimensionality, we can apply Dynamic Programming (DP) methods to solve MDPs by keeping track of Q-values during calculations, thanks to Bellman recursions.

**Value iteration** (so-called backward induction) involves iteratively applying {{< katex >}}\mathcal{T}^*{{< /katex >}} to arbitrarily initialized values {{< katex >}}Q^0{{< /katex >}} until convergence. According to Corollary <a href="#them:fixpoint" data-reference-type="ref" data-reference="them:fixpoint">1</a> and Theorem <a href="#them:fundamental" data-reference-type="ref" data-reference="them:fundamental">1</a>, value iteration converges to {{< katex >}}Q^*{{< /katex >}} as {{< katex >}}k \to \infty{{< /katex >}}, then an optimal policy {{< katex >}}\pi^*{{< /katex >}} can be derived by greedifying {{< katex >}}Q^*{{< /katex >}}.

**Policy iteration** starts with an arbitrary policy {{< katex >}}\pi^0{{< /katex >}} and values {{< katex >}}Q^0{{< /katex >}}. In each iterative step {{< katex >}}k{{< /katex >}}, {{< katex >}}Q^{\pi, k}{{< /katex >}} is calculated by applying Bellman Operator {{< katex >}}\mathcal{T}^{\pi, k}{{< /katex >}} that follows current policy {{< katex >}}{\pi^k}{{< /katex >}} to {{< katex >}}Q^{\pi, {k-1}}{{< /katex >}} from the last iteration, and then {{< katex >}}\pi^{k+1}{{< /katex >}} is derived from greedifying {{< katex >}}Q^{\pi, k}{{< /katex >}}. This process is repeated until convergence, and policy iteration can produce optimal policy after sufficient iterations.

## Bellman Residual for Learning

### Look-up Table

When the transition model is unavailable (model-free), we can use the residuals (RHS minus LHS) of the Bellman Equations as learning objective, {{< katex display=true >}}
\begin{aligned}
    (\mathcal{B}^\pi\circ Q) (s,a) &\doteq  r + \gamma Q(s', \pi(s')) - Q(s, a),\\
    (\mathcal{B}^*\circ Q) (s,a) &\doteq  r + \gamma \max_{a'} Q(s', a') - Q(s, a).
\end{aligned}
{{< /katex >}}
 Assuming that our sampling and parameter updating roughly follow the true state distribution {{< katex >}}\mu(s){{< /katex >}}, the expectation of Bellman residual will be closed to zero at the optima. This approach is often called temporal difference (TD) learning.

In **TD-learning** with learning rate $\alpha$, the update rule for Q-values is, {{< katex display=true >}}
Q(s, a) \leftarrow Q(s, a) + \alpha (\mathcal{B}^\pi\circ Q) (s,a). \label{eq:td-learning}
{{< /katex >}}
 According to Stochastic Approximation Theorem, let {{< katex >}}k{{< /katex >}} be the visitation times of state-action pair, and learning rates {{< katex >}}0 \leqslant \alpha^k < 1{{< /katex >}} satisfies {{< katex >}}\forall (s, a){{< /katex >}}, {{< katex >}}\sum_{k=1}^\infty \alpha^k(s, a) = \infty,\sum_{k=1}^\infty [\alpha^k(s, a)]^2 < \infty{{< /katex >}}. Following TD-learning updates, {{< katex >}}Q^{\pi, k}(s, a){{< /katex >}} converges to {{< katex >}}Q^*(s, a){{< /katex >}} as {{< katex >}}k \to \infty{{< /katex >}} ((Jaakkola, Jordan, and Singh 1994)).

In **Q-learning** that relies on optimal Bellman Equation, the Q-value update is, {{< katex display=true >}}
Q(s, a) \leftarrow Q(s, a) + \alpha (\mathcal{B}^*\circ Q) (s,a). \label{eq:q-learning}
{{< /katex >}}
 According to Stochastic Approximation Theorem, let {{< katex >}}k{{< /katex >}} be the visitation times of state-action pair, and learning rates {{< katex >}}0 \leqslant \alpha^k < 1{{< /katex >}} satisfies {{< katex >}}\forall (s, a){{< /katex >}}, {{< katex >}}\sum_{k=1}^\infty \alpha^k(s, a) = \infty, \sum_{k=1}^\infty [\alpha^k(s, a)]^2 < \infty{{< /katex >}}. Following Q-learning updates, {{< katex >}}Q^{*, k}(s, a){{< /katex >}} converges to {{< katex >}}Q^*(s, a){{< /katex >}} as {{< katex >}}k \to \infty{{< /katex >}} ((Watkins and Dayan 1992)). The deep version of Q-learning algorithm, Deep Q-Network (DQN), is shown in Appendix.

However, the nice property of convergence only holds in the tabular case and cannot be extended to a function approximation as discussed later.

### Function Approximation

To introduce generalization to the value function, we represent the approximated Q-value in a parameterized functional form. Our goal is to minimize the mean squared value error, {{< katex display=true >}}
\mathcal{L}(\theta) = \frac{1}{2}\sum_{s \in \mathcal{S}} \mu(s) \Big[ Q^\text{target} - Q_\theta(s, a) \Big]^2,
{{< /katex >}}
 where {{< katex >}}Q^\text{target}{{< /katex >}} is the ground truth and {{< katex >}}Q_\theta{{< /katex >}} is the prediction. Just like TD-learning, the Bellman residual can be applied for the value function approximation.

##### Semi gradient for Bellman Residual

Similar to stochastic gradient methods with unbiased target estimators, if we use the Bellman Equation to get target Q-value $Q^\text{target}$, but here we just ignore its potential gradient change, the gradient ascent for Bellman residual is, {{< katex display=true >}}
\begin{aligned}
    \Delta_\text{semi} \theta &= -\frac{1}{2}\alpha \nabla_\theta  \Big[Q^\text{target} - Q_\theta(s, a) \Big]^2 \\ 
    &= \alpha \Big[Q^\text{target} - Q_\theta(s, a) \Big] \nabla_\theta Q_\theta(s, a), \text{ where } Q^\text{target} = r + \gamma Q_{\textcolor{red}{\theta}}(s', a')\label{eq:semi-grad}
\end{aligned}
{{< /katex >}}
 Since we neglects a part of the gradient of {{< katex >}}Q^\text{target}{{< /katex >}}, it is called semi gradient for Bellman residual ({{< katex >}}\theta{{< /katex >}} in red). Though semi-gradient methods are fast and simple, they could have divergence issue, e.g., Baird’s counter-example (the star problem).

##### Full Gradient for Bellman Residual

The full Bellman residual gradient should include all gradient components, including the gradient of the target estimation, {{< katex display=true >}}
\begin{aligned}
    \Delta_\text{full} \theta &= -\frac{1}{2}\alpha \nabla_\theta  \Big[ r + \gamma Q_\theta(s', a') - Q_\theta(s, a) \Big]^2 \\
    & = -\alpha \Big[ r + \gamma Q_\theta(s', a') - Q_\theta(s, a) \Big] \Big[ \gamma\nabla_\theta Q_\theta(s', a') - \nabla_\theta Q_\theta(s, a) \Big].
\end{aligned}
{{< /katex >}}
 If the approximation system is general enough and the value functions are continuous, the full Bellman residual gradient is guaranteed to converge to the optima. However, this is at the sacrifice of learning speed, as illustrated by the hall problem.

##### Hybrid Gradient for Bellman Residual

In contrast to Figure <a href="#subfig:sg-increase" data-reference-type="ref" data-reference="subfig:sg-increase">1</a> where $\Delta_\text{semi}$ boosts $\Delta_\text{full}$, Figure <a href="#subfig:sg-decrease" data-reference-type="ref" data-reference="subfig:sg-decrease">3</a> represents the case where the semi gradient may diverge. (Baird 1995) combined these 2 methods: to keep stable, $\Delta_\text{B}$ should stay in the same direction as $\Delta_\text{full}$ (above the perpendicular axis); meanwhile, $\Delta_\text{B}$ should stay as close as possible to $\Delta_\text{semi}$ to increase learning speed. {{< katex display=true >}}
\begin{aligned}
    \Delta_\text{B} \theta &= (1 - \omega)  \cdot \Delta_\text{semi}\theta + \omega \cdot \Delta_\text{full}\theta, \\
    &=-\alpha \Big[ r + \gamma Q_\theta(s', a') - Q_\theta(s, a) \Big] \Big[\omega \gamma \nabla_\theta Q_\theta(s', a') - \nabla_\theta Q_\theta(s, a) \Big],\\
    &\text{s.t.,} \ \Delta_\text{B}\theta \cdot \Delta_\text{full}\theta\geqslant 0 \Leftrightarrow \omega \geqslant \frac{\Delta_\text{semi}\theta \cdot \Delta_\text{full}\theta}{\Delta_\text{semi}\theta \cdot \Delta_\text{full}\theta - \Delta_\text{full}\theta \cdot \Delta_\text{full}\theta}.
\end{aligned}
{{< /katex >}}

## References

{{< references >}}
<li>Baird, Leemon C. 1995. “Residual Algorithms: Reinforcement Learning with Function Approximation.” In <em>Machine Learning Proceedings 1995</em>, 30–37. Elsevier.</li>
<li>Bellman, Richard. 1957. <em>Dynamic Programming</em>. Princeton, NJ: Princeton University Press.</li>
<li>Jaakkola, Thomas, Michael I. Jordan, and Satinder P. Singh. 1994. “On the Convergence of Stochastic Iterative Dynamic Programming Algorithms.” <em>Neural Computation</em> 6 (6): 1185–1201. <https://doi.org/10.1162/neco.1994.6.6.1185>.</li>
<li>Sutton, Richard S., and Andrew G. Barto. 2018. <em>Reinforcement Learning: An Introduction</em>. 2nd ed. MIT Press.</li>
<li>Szepesvári, Csaba. 2010. <em>Algorithms for Reinforcement Learning</em>. Vol. 4. Synthesis Lectures on Artificial Intelligence and Machine Learning 1. Morgan & Claypool Publishers. <https://doi.org/10.2200/S00268ED1V01Y201005AIM009>.</li>
<li>Watkins, Christopher J. C. H., and Peter Dayan. 1992. “Q-Learning.” <em>Machine Learning</em> 8 (3–4): 279–92. <https://doi.org/10.1007/BF00992698>.</li>
{{< /references >}}



<!-- footnote converted to hint above -->
<!-- migrated from leaf-bundle to single-file naming -->
<!-- moved to root content -->
<!-- moved back under rl/ -->
