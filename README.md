# LovelyBuggies.github.io

This site serves my homepage and a Hugo Book–based Blog under `/notes/`.

Quick start
- Local preview: `cd notes && hugo server -D`, open `http://localhost:1313/notes/`.
- Build site: `cd notes && hugo` (outputs to `public/` inside `notes`). The CI workflow builds into `dist/notes`.
- CI deploy (optional): set Pages Source to “GitHub Actions” and push to `main`; the workflow builds and deploys.

Add a new blog post
1) Create a single file under one of the sections, e.g. `notes/content/rl/my_new_post.md`, `notes/content/rl/marl/my_new_post.md`, or `notes/content/large-language-models/my_new_post.md`. Name it from the title with underscores.
2) Add front matter, e.g.:

```
---
title: "My Post Title"
date: 2025-01-10
weight: 10
# Optional if the page uses math:
# math: true
---

Your content in Markdown here.
```

- Ordering in sidebar: lower `weight` appears higher.
- Assets: store images under `/imgs/blog/<post-slug>/` and link with absolute paths, e.g. `![fig](/imgs/blog/my_new_post/fig1.png)`.
- Math: use `$...$` (inline) and `$$...$$` (block). Add `math: true` to the page front matter.

Deployment options
- Branch-based (no CI): Settings → Pages → Source = “Deploy from a branch”, Branch = `main`, Folder = `/`.
- GitHub Actions: Settings → Pages → Source = “GitHub Actions”. Actions workflow builds and deploys `main`.

Notes
- Section `_index.md` files are kept only to name and order sidebar groups and are not rendered.
- The Blog uses the Hugo Book theme located at `notes/themes/book` (git submodule).
- Site config is in `notes/config.toml`.

Media
- Blog favicon is shared with the homepage: `/imgs/icon/nameicon.svg`.
- Post images are stored under `/imgs/blog/<post-slug>/`.
