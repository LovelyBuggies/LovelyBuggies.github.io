# LovelyBuggies.github.io

This site serves my homepage and a Hugo Book–based Blog under `/notes/`.

Quick start
- Local preview: `cd notes-src && hugo server -D`, open `http://localhost:1313/notes/`.
- Build site: `cd notes-src && hugo` (outputs to `../notes/`). Commit and push the `notes/` folder.
- CI deploy (optional): set Pages Source to “GitHub Actions” and push to `main`; the workflow builds and deploys.

Add a new blog post
1) Create a folder under `notes-src/content/docs/`, e.g. `notes-src/content/docs/my-post/`.
2) Inside it, add an `index.md` (or `_index.md`) with front matter, e.g.:

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
- Recent listing: any page with a `date` appears on “Recent Posts”.
- Assets: place images/files next to your markdown and link relatively, e.g. `![fig](figure.png)`.
- Math: use `$...$` (inline) and `$$...$$` (block). Add `math: true` to the page front matter.

Deployment options
- Branch-based (no CI): Settings → Pages → Source = “Deploy from a branch”, Branch = `main`, Folder = `/`.
- GitHub Actions: Settings → Pages → Source = “GitHub Actions”. Actions workflow builds and deploys `main`.

Notes
- The Blog uses the Hugo Book theme located at `notes-src/themes/book` (git submodule).
- Site config is in `notes-src/config.toml`.
