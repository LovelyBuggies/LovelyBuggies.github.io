---
title: "How to Add a Post"
weight: 3
---

Add a new blog post
- Create a folder under `notes-src/content/docs/`, e.g., `notes-src/content/docs/my-post/`.
- Inside, add `index.md` with front matter and content. Example:

```markdown
---
title: "My Post Title"
date: 2025-01-10
weight: 10
# Enable if the page uses math
# math: true
---

Your content in Markdown here.
```

Notes
- Sidebar order: smaller `weight` shows higher.
- Recent: any page with a `date` appears on lists you add later.
- Assets: put images next to your markdown and link relatively, e.g. `![fig](figure.png)`.
- Math: use `$...$` (inline), `$$...$$` (block), add `math: true` and `{{< katex />}}` near top.

Build and publish
- Local preview: `cd notes-src && hugo server -D` → http://localhost:1313/notes/
- Build: `cd notes-src && hugo` → outputs to `../notes/`.
- Commit and push the updated `notes/` folder (or let CI build if you use Actions).

Legacy note
- Older instructions remain available at `/archive/posts/instructions.html`.
