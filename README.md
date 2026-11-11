# LovelyBuggies.github.io

The academic homepage is static HTML. **Shuo's Blog** uses Hugo Extended 0.150.0. English articles keep their original paths under `/notes/`; Simplified Chinese articles have the same path after `/notes/zh/`.

## Write and preview

Write the English article in `notes/content/lm/`, `notes/content/rl/`, or `notes/content/rl/marl/`. The sections supply the topic tags LLM, RL and Multi-Agent RL. A MARL article also appears under RL. Tag archives and the main list use the original publication date, newest first.

```yaml
---
title: "Article title"
date: 2026-01-10
description: "A short summary for the index."
postType: review
readingTime: 15
---
```

Start the body with the introduction. The template adds the title, author, publication date and tags. The optional `readingTime` overrides Hugo's estimate in minutes. Existing `postType` values are `review` and `thought`. Headings from level two onward populate the contents menu.

Link author–year citations to the original works where they appear in the text. Inline math uses `$...$`; display math uses `$$...$$`. Hugo compiles these and the existing `katex` shortcodes into MathML at build time. Keep figures under `notes/assets/imgs/` and reference them as `![Description](/imgs/path/to/figure.png)`.

Run `make preview` and open `http://localhost:1313/notes/`. Run `make build` to generate the blog under `dist/notes/`.

## Automatic translation and publication

A push to `main`, or a manual run of **Deploy Pages**, builds English first to obtain the actual heading anchors. The workflow uses the pinned [CoMLRL documentation translation engine](https://github.com/OpenMLRL/CoMLRL/blob/27f309ede6857b453ba8d1c984d6ff42aa32c92a/docs/scripts/translations.py) with Ollama and `translategemma:4b` on the GitHub-hosted runner. Translation requires no external model service or API key. The repository's Pages source should be **GitHub Actions**.

The runner caches translated text. When all text is current, the build reconstructs the Chinese articles from the cache without starting Ollama. A change to a paragraph or title translates the missing text and reuses the rest. The first run downloads the model and translates every article. Dates, tag identifiers, equations, code, citation lists, and link destinations come from the English source. Language changes keep readers on the corresponding article or tag.

Generated Chinese article Markdown exists only in the build workspace. Before uploading the Pages artifact, the workflow checks both languages, matching article URLs, anchors and compiled math. An incomplete translation stops the deployment. Correct the source or rerun the workflow to retry; completed translations can be reused. Interface strings and section metadata are maintained in `notes/i18n/` and the localized section indexes.

To exercise the same translation process locally, check out the CoMLRL revision pinned in the workflow and use its path below. Start Ollama and run `ollama pull translategemma:4b` if the model is not already installed.

```sh
english_site=$(mktemp -d)
HUGO_DISABLELANGUAGES=zh hugo -s notes -d "$english_site"
python3 scripts/translate_blog.py --engine-root ../CoMLRL --english-site "$english_site"
make build
python3 scripts/check_blog_languages.py dist/notes
```

Use `--cache-only` on the translation command to rebuild from a complete cache. Local generated article siblings and the translation cache are Git-ignored.
