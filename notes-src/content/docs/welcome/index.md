---
title: "Welcome"
description: "Starter post using Hugo Book in Markdown."
tags: ["getting-started"]
---

# Welcome

This is a fresh Markdown post rendered with the Hugo Book theme. Use it as a template for your future notes and posts.

## What’s Included

- Sections and ToC are automatic based on headings.
- KaTeX math support is enabled: $V(s) = \max_a Q(s,a)$.
- Syntax-highlighted code blocks:

```python
def value(s, Q):
    return max(Q[s].values())
```

## Tips

{{< hint info >}}
Place images next to this file, e.g., `docs/welcome/diagram.png`, and reference with `![Alt](diagram.png)`.
{{< /hint >}}

- Add front matter fields like `tags`, `weight`, or `draft: true` as needed.
- For math blocks, wrap in `$$ ... $$` for display equations.

## Next Steps

Create your own post by copying this folder and editing the front matter and content. Example:

```
notes-src/content/docs/<section>/<my-post>/index.md
```

Set a descriptive `title` and `description`, and start writing in Markdown.

