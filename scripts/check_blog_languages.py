"""Check the rendered English and Chinese blog before publishing the Pages artifact."""

import argparse
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit

from translate_blog import technical_label


class Page(HTMLParser):
    """Collect publishing invariants from Hugo HTML, including minified output."""

    def __init__(self, path):
        super().__init__()
        self.lang = self.canonical = self.switch = self.title = ""
        self.article = False
        self.math = []
        self.code = []
        self.math_errors = False
        self.in_code = False
        self.ids = []
        self.headings = []
        self.resources = []
        self.links = []
        self.all_links = []
        self.prose_text = []
        self.prose_depth = 0
        self.in_title = self.in_annotation = self.in_heading = False
        self.skip_text = 0
        self.reference_lists = 0
        self.feed(path.read_text(encoding="utf-8"))
        self.close()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = attrs.get("class", "").split()
        if tag == "html":
            self.lang = attrs.get("lang", "")
        if tag == "link" and "canonical" in attrs.get("rel", "").split():
            self.canonical = attrs.get("href", "")
        if tag == "a" and "href" in attrs:
            self.all_links.append(attrs["href"])
            if "language-switch" in classes:
                self.switch = attrs["href"]
        if tag == "h1":
            self.in_title = True
        if tag == "div" and (self.prose_depth or "prose" in classes):
            self.prose_depth += 1
            self.article = True
        if not self.prose_depth:
            return
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.in_heading = True
            self.headings.append((tag, attrs.get("id")))
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if tag == "merror" or "katex-error" in classes:
            self.math_errors = True
        if tag == "code":
            self.in_code = True
            self.code.append("")
        if tag == "annotation" and attrs.get("encoding") == "application/x-tex":
            self.in_annotation = True
            self.math.append("")
        if tag == "ol" and (self.reference_lists or "article-references" in classes):
            self.reference_lists += 1
        if tag in ("math", "code", "pre"):
            self.skip_text += 1
        if tag == "a" and "href" in attrs:
            self.links.append(attrs["href"])
        if tag in ("img", "source"):
            self.resources.append((tag, attrs.get("src", ""), attrs.get("srcset", "")))

    def handle_endtag(self, tag):
        if tag == "h1":
            self.in_title = False
        if not self.prose_depth:
            return
        if tag == "code":
            self.in_code = False
        if tag == "div":
            self.prose_depth -= 1
        elif tag == "ol" and self.reference_lists:
            self.reference_lists -= 1
        elif tag == "annotation":
            self.in_annotation = False
        elif tag in ("math", "code", "pre"):
            self.skip_text -= 1
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.in_heading = False

    def handle_data(self, text):
        if self.in_title:
            self.title += text
        if self.in_annotation:
            self.math[-1] += text
        if self.in_code:
            self.code[-1] += text
        if self.prose_depth and not self.skip_text and not self.in_heading and not self.reference_lists:
            self.prose_text.append(text)


def translated_title(en, zh):
    en, zh = en.strip(), zh.strip()
    return bool(re.search(r"[\u3400-\u9fff]", zh) or (
        zh == en and technical_label(en)
    ))


def translated_prose(en, zh):
    # An otherwise-English page with a translated heading or notice is not a translation.
    def passages(page):
        return {" ".join(text.split()) for text in page.prose_text if len(text.split()) >= 10}

    original = passages(en)
    unchanged = original.intersection(passages(zh))
    return bool(re.search(r"[\u3400-\u9fff]", "".join(zh.prose_text))) and (
        not original or len(unchanged) * 2 < len(original)
    )


def compare(output):
    pages = {path.parent.relative_to(output): Page(path) for path in output.rglob("index.html")}
    articles = {p: page for p, page in pages.items() if page.article and p.parts[:1] != ("zh",)}
    if not articles:
        raise ValueError(f"No English articles found in {output}")
    issues = []
    localizations = {
        page.canonical: pages[Path("zh") / path].canonical for path, page in pages.items()
        if path.parts[:1] != ("zh",) and Path("zh") / path in pages
    }
    for path in [Path("."), *articles]:
        zh = pages.get(Path("zh") / path)
        en = pages.get(path)
        if en is None or zh is None:
            issues.append(f"{path}: English or Chinese counterpart missing")
            continue
        if not en.canonical or not zh.canonical:
            issues.append(f"{path}: missing canonical URL")
            continue
        if not (en.lang.lower().startswith("en") and zh.lang.lower().startswith("zh")):
            issues.append(f"{path}: wrong document language")
        if (not en.switch or not zh.switch
                or urljoin(en.canonical, en.switch) != zh.canonical
                or urljoin(zh.canonical, zh.switch) != en.canonical):
            issues.append(f"{path}: language switch does not lead to the matching page")

    def chinese_target(url):
        parts = urlsplit(url)
        origin = urlunsplit(parts._replace(query="", fragment=""))
        return urlunsplit(urlsplit(localizations.get(origin, origin))._replace(
            query=parts.query, fragment=parts.fragment
        ))

    total_math = 0
    for path, en in articles.items():
        zh = pages.get(Path("zh") / path)
        if zh is None:
            continue
        for label, matches in (
            ("Chinese title", translated_title(en.title, zh.title)),
            ("Chinese body prose", translated_prose(en, zh)),
            ("MathML source expressions", en.math == zh.math and not en.math_errors and not zh.math_errors),
            ("code samples", en.code == zh.code),
            ("heading levels and anchors", en.headings == zh.headings),
            ("element IDs, including formula anchors", en.ids == zh.ids),
            ("images and media sources", en.resources == zh.resources),
            ("article links", [chinese_target(urljoin(en.canonical, href)) for href in en.links]
                == [urljoin(zh.canonical, href) for href in zh.links]),
        ):
            if not matches:
                issues.append(f"{path}: {label} missing or different")
        total_math += len(en.math)
    for folder in (Path("."), Path("zh")):
        home = pages.get(folder)
        if home:
            listed = {urljoin(home.canonical, href) for href in home.all_links}
            for path in articles:
                page = pages.get(folder / path)
                if page and page.canonical not in listed:
                    issues.append(f"{folder}: archive is missing {page.canonical}")
    if issues:
        raise ValueError("\n".join(issues))
    print(f"Checked {len(articles)} English/Chinese article pairs and {total_math} math expressions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("public", type=Path, help="built Hugo output, e.g. dist/notes")
    args = parser.parse_args()
    try:
        compare(args.public)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
