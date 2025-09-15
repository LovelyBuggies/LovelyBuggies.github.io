---
title: "How to Add a Blog/Note"
weight: 5
---

This site uses Hugo Book. Each note is just a Markdown page under `content/docs/`.

Quick start
- Create a new folder for the note (a "leaf bundle"): `content/docs/my-new-note/`
- Inside it, add `_index.md` with front matter and your content.
- Run `hugo server -D` to preview at `http://localhost:1313/notes/`.

Minimal example

```text
notes-src/
  content/
    docs/
      my-new-note/
        _index.md
```

`_index.md` template

```markdown
---
title: "My New Note"
date: 2025-01-01
weight: 15
---

Write your content here in Markdown. Use headings to populate the table of contents.

## A Section

Details...
```

Tips
- Ordering: smaller `weight` shows higher in the sidebar.
- Grouping: create subfolders to nest sections, e.g., `docs/rl/ppo/_index.md`.
- Assets: place images or files alongside `_index.md` and reference them relatively, e.g. `![img](figure.png)`.
- Code blocks: fenced triple backticks ``` with a language name for syntax highlighting.
- Dates: if you want a chronological list, you can add a section index page that ranges over pages by date; otherwise the sidebar order is by `weight`.

Math
- Use `$...$` for inline math and `$$...$$` for block equations.
- KaTeX assets are included by the theme; nothing extra is needed per page.
- See the "Math Demo" page for examples.

Build and publish
- From `notes-src`: run `hugo` to generate static pages into `../notes/`.
- Commit `notes/` so GitHub Pages serves `https://lovelybuggies.github.io/notes/`.
