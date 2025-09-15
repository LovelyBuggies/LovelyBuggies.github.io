---
title: "Math Demo"
date: 2025-01-02
weight: 20
math: true
---

{{< katex />}}

Inline math like $a^2 + b^2 = c^2$ (also works with \(a^2 + b^2 = c^2\)) and display math:

$$
\nabla_\theta \, \mathbb{E}_{\pi_\theta}[R] 
= \mathbb{E}_{\pi_\theta} [ \, \nabla_\theta \log \pi_\theta(a\mid s) \, R \, ]
$$

If this renders as formatted equations, KaTeX is working.

Notes
- Use `$...$` for inline, `$$...$$` or `\[ ... \]` for block equations.
- `math: true` in front matter enables math scripts for this page.
