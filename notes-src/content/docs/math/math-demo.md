---
title: "Math Demo"
date: 2025-01-02
weight: 50
math: true
---

Inline math like $a^2 + b^2 = c^2$ and display math:

$$
\nabla_\theta \, \mathbb{E}_{\pi_\theta}[R] 
= \mathbb{E}_{\pi_\theta} [ \, \nabla_\theta \log \pi_\theta(a\mid s) \, R \, ]
$$

If this renders as formatted equations, KaTeX is working.

Notes
- Use `$...$` for inline, `$$...$$` for block equations.
- No extra escaping is needed inside code blocks; use Markdown fences for code, math fences for equations.
