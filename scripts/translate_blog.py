#!/usr/bin/env python3
"""Build Hugo .zh.md article siblings using CoMLRL's Ollama translator."""

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
from html.parser import HTMLParser
import importlib.util
import json
from pathlib import Path
import re
import sqlite3
import sys
from tempfile import NamedTemporaryFile


MODEL = "translategemma:4b"
ENGINE_REV = "27f309ede6857b453ba8d1c984d6ff42aa32c92a"
HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
DISPLAY = re.compile(r"^(title|description|linkTitle):[ \t]*(.+)$", re.M)
TOKEN = re.compile(r"ZXQ(\d{5})QXZ")
SURNAME = r"[A-ZÀ-Þ][\wÀ-ž’'-]*(?: (?:de|van|von|der|den|la) [A-ZÀ-Þ][\wÀ-ž’'-]*)?"
AUTHORS = rf"{SURNAME}(?: et\.? al\.?|(?:, {SURNAME})*(?:,? (?:and|&) {SURNAME})?)"
YEAR = r"(?:18|19|20)\d{2}[a-z]?"
LOCATOR = r"(?:, (?:Sec\.|pp?\.|Table|Theorem) [\d.IVXLCDM]+)?"
CITATION = (
    rf"\b{AUTHORS}(?:,? {YEAR}{LOCATOR}"
    rf"| \({YEAR}(?:[,;] revised {YEAR})?{LOCATOR}\))"
)
# These additional spans are specific to technical articles. CoMLRL's scanner
# already handles fenced/inline code, KaTeX, $math$, links, and quoted HTML.
EXTRA = re.compile(
    r"\{\{[<%]\s*references\s*[>%]\}\}[\s\S]*?\{\{[<%]\s*/references\s*[>%]\}\}"
    r"|\{\{[<%]\s*katex\s*/\s*[>%]\}\}|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)"
    r"|<(?P<raw>pre|code|script|style)\b[^>]*>[\s\S]*?</(?P=raw)\s*>"
    r"|(?m:^(?: {4}|\t)[^\n]+)|\\[^\s\w]|\[\^[^\]\n]+\]|"
    + CITATION
    + r"|&(?:#[xX]?[\da-fA-F]+|[A-Za-z][\w]+);|[\[\]|*_~`{}]"
)
PREFIX = re.compile(r"^([ \t]*(?:(?:#{1,6}|[-+*]|\d+[.)]|>)[ \t]+)*)")
LABEL = re.compile(
    r'(?P<prefix>\b(?:title|alt|caption|label|description)="|\b(?:tab|details)\s+")'
    r'(?P<label>[^"\n]+)"'
)
GENERATED = "translationGenerated: true"
# Technical identifiers and versioned model names retain their published spelling.
TECHNICAL_ID = re.compile(
    r"(?:(?:[A-Z][a-z]+-)?[A-Z][A-Z0-9]+(?:-[A-Z0-9]+)*"  # Dec-POMDP, GPT-4
    r"|(?:Llama|Gemma|Qwen|Mistral|Gemini|Claude|Phi)[ -]\d+(?:\.\d+)*"
    r"|[A-Z][a-z]+[A-Z][A-Za-z0-9]*|[A-Z]-[a-z]+)[:.]?"     # PettingZoo, Q-learning
)


def technical_label(value):
    """Preserve model/algorithm names, including table labels such as MMDP (Cen-MMDP)."""
    value = value.strip()
    if TECHNICAL_ID.fullmatch(value):
        return True
    parts = re.findall(r"[^\s(),;/+]+", value)
    return bool(parts) and all(TECHNICAL_ID.fullmatch(part) for part in parts)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def is_generated(path, engine):
    front = engine.FRONT.match(path.read_text(encoding="utf-8"))
    return bool(front and GENERATED in front[1].splitlines())


def load_engine(root):
    script = root / "docs/scripts/translations.py"
    spec = importlib.util.spec_from_file_location("comlrl_translations", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProseHeadings(HTMLParser):
    """Read the actual identifiers produced by Goldmark, including duplicates."""

    def __init__(self):
        super().__init__()
        self.depth = 0
        self.headings = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "div":
            if self.depth or "prose" in attrs.get("class", "").split():
                self.depth += 1
        if self.depth and re.fullmatch("h[1-6]", tag):
            self.headings.append((int(tag[1]), attrs["id"]))

    def handle_endtag(self, tag):
        if tag == "div" and self.depth:
            self.depth -= 1


def anchor_headings(source, english_html, engine):
    parsed = ProseHeadings()
    parsed.feed(english_html.read_text(encoding="utf-8"))
    headings = engine.heading_lines(source)
    if [level for _, level, _ in headings] != [level for level, _ in parsed.headings]:
        raise ValueError(
            f"English HTML does not match the source headings: {english_html}"
        )
    lines = source.splitlines(keepends=True)
    for (line, _, heading), (_, anchor) in zip(headings, parsed.headings):
        if not re.search(r"\{#[^}]+\}\s*$", heading):
            lines[line] = lines[line].rstrip() + " {#" + anchor + "}\n"
    return "".join(lines)


@dataclass(frozen=True)
class Text:
    value: str
    mode: str = "prose"
    literals: tuple = ()


# Keep inline labels intact so prose cannot move inside a link or theorem label.
LINK = re.compile(
    r'(?P<html><(?P<tag>a|strong|em|b|i)\b(?:[^>"\']|"[^"]*"|\'[^\']*\')*>)(?P<html_label>[^<>\n]+)</(?P=tag)>'
    r"|!?\[(?P<md_label>[^\[\]\n]+)\]\(",
    re.I,
)


@dataclass(frozen=True)
class InlineLink:
    opening: str
    label: str
    closing: str

    def field(self):
        # Citations and names of linked tools already identify their target.
        if (
            not re.search(r"[A-Za-z]", self.label)
            or re.fullmatch(CITATION, self.label)
            or re.fullmatch(rf"{SURNAME} et al\.", self.label)
            or (
                re.fullmatch(r"[\w.-]+", self.label)
                and self.label.casefold() in (self.opening + self.closing).casefold()
            )
        ):
            return None
        return Text(self.label, "link")


def inline_links(body):
    for m in LINK.finditer(body):
        name = "html_label" if m["html"] else "md_label"
        label = m[name]
        if re.search(r"[<>_$*`{}\\]", label):
            continue  # nested formatting uses the ordinary protected-span scanner
        end = m.end()
        if not m["html"]:
            depth = 1
            while end < len(body) and body[end] != "\n" and depth:
                c = body[end]
                end += 1
                if c == "\\":
                    end += 1
                elif c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
            if depth:
                continue
        yield (m.start(), end), InlineLink(
            body[m.start() : m.start(name)], label, body[m.end(name) : end]
        )


def _slots(source, engine):
    """Keep the original document as a template. Only Text slots are translated."""
    front = engine.FRONT.match(source)
    if not front:
        raise ValueError("Expected YAML front matter")
    start = 0
    for match in DISPLAY.finditer(front[0]):
        yield front[0][start : match.start(2)]
        val = match[2].strip()
        if val.startswith('"'):
            val = json.loads(val)
        elif val.startswith("'") and val.endswith("'"):
            val = val[1:-1].replace("''", "'")
        elif val.startswith(("|", ">")):
            raise ValueError("Use a single-line YAML title or description")
        yield Text(val, "yaml")
        start = match.end(2)
    yield front[0][start : front[0].rfind("---")] + GENERATED + "\n---\n"

    body = source[front.end() :]
    links = dict(inline_links(body))
    spans = [(m.start(), m.end()) for m in engine.PROTECTED.finditer(body)] + list(
        links
    )
    spans += [m.span() for m in EXTRA.finditer(body)]
    spans += [
        (m.start(1), m.end(1)) for m in re.finditer(PREFIX.pattern, body, re.M) if m[1]
    ]
    # Prefer the outermost span when tokens overlap (e.g. KaTeX inside a proof).
    offset = 0
    for begin, end in sorted(spans, key=lambda pair: (pair[0], -pair[1])):
        if begin < offset:
            continue
        yield from prose(body[offset:begin])
        raw = body[begin:end]
        if (begin, end) in links:
            yield links[begin, end]
        elif re.match(r"\{\{[<%]\s*(?:badge|image|tab|details)\b", raw):
            mark = 0
            for m in LABEL.finditer(raw):
                yield raw[mark : m.start("label")]
                yield Text(m["label"], "attribute")
                mark = m.end("label")
            yield raw[mark:]
        else:
            yield raw
        offset = end
    yield from prose(body[offset:])


def prose(chunk):
    for line in chunk.splitlines(keepends=True):
        if not re.search(r"[A-Za-z]", line) or technical_label(line):
            yield line
            continue
        m = re.fullmatch(r"(\s*)(.*?)(\s*)", line, re.S)
        yield m[1]
        yield Text(m[2])
        yield m[3]


def pack_line(pieces):
    """Translate the prose of one Markdown line in context, between immutable tokens."""
    if not any(isinstance(p, Text) for p in pieces):
        yield "".join(pieces)
        return
    runs = []
    for piece in pieces:
        if isinstance(piece, str) and runs and isinstance(runs[-1], str):
            runs[-1] += piece
        else:
            runs.append(piece)
    mode = "prose"
    if isinstance(runs[0], str):
        match = PREFIX.match(runs[0])
        prefix = match[0]
        if prefix.lstrip().startswith("#"):
            mode = "heading"
        yield prefix
        runs[0] = runs[0][len(prefix) :]
    suffix = ""
    if isinstance(runs[-1], str):
        anchor = re.search(r"\s+\{#[^}]+\}\s*$", runs[-1])
        if anchor:
            suffix = anchor[0]
            runs[-1] = runs[-1][: anchor.start()]
    text, literals = [], []
    for run in runs:
        if isinstance(run, Text):
            text.append(run.value)
        elif re.fullmatch(r"[\s.,:;!?…-]*", run):
            text.append(run)
        else:
            match = re.fullmatch(r"(\s*)(.*?)(\s*)", run, re.S)
            index = len(literals)
            literals.append(match[2])
            text.extend((match[1], f"ZXQ{index:05d}QXZ", match[3]))
    yield Text("".join(text), mode, tuple(literals))
    yield suffix


def slots(source, engine, *, db=None, options=None):
    tokens = list(_slots(source, engine))
    labels = [p.field() for p in tokens if isinstance(p, InlineLink) and p.field()]
    translated = fill_cache(labels, db, options, engine) if db is not None else {}
    line = []
    for piece in tokens:
        if isinstance(piece, InlineLink):
            label = translated.get(piece.field())
            piece = (
                piece.opening
                + (escaped_prose(label) if label else piece.label)
                + piece.closing
            )
        if isinstance(piece, Text) and piece.mode == "prose":
            line.append(piece)
        elif isinstance(piece, Text) or "\n" in piece or piece == "|":
            # A punctuation-only line ending belongs to the preceding sentence.
            # Multi-line formulas/fences, by contrast, are immutable blocks.
            if isinstance(piece, str):
                tail = re.match(r"^[ \t.,:;!?…-]+(?=\n)", piece)
                if tail:
                    line.append(tail[0])
                    piece = piece[tail.end() :]
            yield from pack_line(line)
            line = []
            yield piece
        else:
            line.append(piece)
    yield from pack_line(line)


def escaped_prose(value):
    return re.sub(r"[<>&\[\]*_`~|$\\{}]", lambda m: f"&#{ord(m[0])};", value)


def restore(unit, translated):
    """Only original literals can produce markup; the model supplies plain prose."""
    pieces = []
    position = 0
    for token in TOKEN.finditer(translated):
        prose = translated[position : token.start()]
        # Preserve Markdown flanking rules (e.g. **emphasis**, [link], inline code).
        original = unit.value.index(token[0])
        if original and not unit.value[original - 1].isspace():
            prose = prose.rstrip()
        if position and not unit.value[previous_end : previous_end + 1].isspace():
            prose = prose.lstrip()
        pieces.extend([escaped_prose(prose), unit.literals[int(token[1])]])
        position = token.end()
        previous_end = original + len(token[0])
    rest = translated[position:]
    if position and not unit.value[previous_end : previous_end + 1].isspace():
        rest = rest.lstrip()
    pieces.append(escaped_prose(rest))
    return "".join(pieces)


def render(parts, translations):
    result = []
    for p in parts:
        if isinstance(p, str):
            result.append(p)
            continue
        value = translations[p]
        if p.mode == "yaml":
            result.append(json.dumps(value, ensure_ascii=False))
        elif p.mode == "attribute":
            result.append(value.replace('"', "”"))
        else:
            result.append(restore(p, value))
    return "".join(result)


def validate_text(unit, target):
    if (
        not isinstance(target, str)
        or not target.strip()
        or any(c in target for c in ("\n", "\r", "\x00", "\ufffd"))
    ):
        raise ValueError("empty or invalid translated text")
    if TOKEN.findall(target) != TOKEN.findall(unit.value):
        raise ValueError("missing, reordered or duplicated Markdown placeholders")
    plain = TOKEN.sub("", unit.value).strip()
    output = TOKEN.sub("", target).strip()
    # A response can contain every placeholder AND echo the raw link or formula.
    # In Hugo that can become a duplicate link or a visible, uncompiled shortcode.
    markup = r"https?://|\\[A-Za-z]+|\{\{[<%]|</?[A-Za-z][\w-]*(?=[\s/>])|\$"
    if Counter(re.findall(markup, output)) - Counter(re.findall(markup, plain)):
        raise ValueError("translation inserted technical markup into the prose")
    if unit.literals and "…" in output and "…" not in plain:
        raise ValueError("translation echoed the context's formula stand-in")
    words = re.findall(r"[A-Za-z]+", plain)
    if unit.mode != "prose" or len(words) >= 3:
        if words and not technical_label(plain) and not HAN.search(output):
            raise ValueError(f"no Chinese translation for {plain[:70]!r}")
    if unit.mode != "prose" and len(output) > max(14, round(1.2 * len(plain))):
        raise ValueError("translated label is unexpectedly long")


def retry_markers(unit, options, engine):
    """Try visually distinct markers; small models can merge adjacent ASCII tokens."""
    markers = {
        f"⟦{chr(65 + i % 26)}{i // 26 or ''}⟧": f"ZXQ{i:05d}QXZ"
        for i in range(len(unit.literals))
    }
    source = TOKEN.sub(lambda m: list(markers)[int(m[1])], unit.value)
    result = engine._request_batch(
        {"s0": source}, endpoint=options.endpoint, model=options.model, language="zh-cn"
    )["s0"]
    for marker, token in markers.items():
        result = result.replace(marker, token)
    return result


def request_fields(options, instruction, fields):
    from urllib.request import Request, urlopen

    schema = {
        "type": "object",
        "properties": {k: {"type": "string"} for k in fields},
        "required": list(fields),
        "additionalProperties": False,
    }
    body = {
        "model": options.model,
        "stream": False,
        "think": False,
        "format": schema,
        "messages": [
            {
                "role": "system",
                "content": instruction,
            },
            {
                "role": "user",
                "content": json.dumps(fields, ensure_ascii=False),
            },
        ],
        "options": {
            "temperature": 0,
            "num_ctx": 8192,
            "num_predict": min(2048, 96 + 3 * sum(map(len, fields.values()))),
        },
    }
    request = Request(
        options.endpoint.rstrip("/") + "/api/chat",
        json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=300) as response:
        reply = json.loads(json.load(response)["message"]["content"])
    if (
        not isinstance(reply, dict)
        or reply.keys() != fields.keys()
        or not all(isinstance(v, str) for v in reply.values())
    ):
        raise ValueError("invalid contextual translation response")
    return reply


def retry_in_context(unit, options, engine):
    """Recover failed placeholder alignment without sending isolated fragments."""
    matches = list(TOKEN.finditer(unit.value))
    segments = re.split(r"ZXQ\d{5}QXZ", unit.value)
    fields = {f"s{i}": s for i, s in enumerate(segments) if re.search(r"[A-Za-z]", s)}
    if not fields:
        raise ValueError("no prose to translate")
    prompt = (
        "Translate the whole English passage into fluent Simplified Chinese. Each key is the prose "
        "between two immutable pieces of Markdown/math in the SAME passage. Return the Chinese "
        "for each prose segment so that the pieces together form one natural translation. Keep "
        "terms and numeral values accurate. A determiner may become empty. Do not repeat prose "
        "in another field. Do not output markup, placeholders, explanations, or extra text."
    )
    reply = request_fields(
        options,
        prompt
        + "\nEnglish source for context: "
        + TOKEN.sub("…", unit.value),
        fields,
    )
    out = []
    for i, original in enumerate(segments):
        out.append(reply.get(f"s{i}", original))
        if i < len(matches):
            out.append(matches[i][0])
    return "".join(out)


def retry_one_segment_at_a_time(unit, options, engine):
    """Last resort for dense math: restore every token and translate only its prose gaps."""
    from http.client import HTTPException

    # Give the linguistic context without feeding long LaTeX expressions back to the model.
    context = TOKEN.sub("…", unit.value)[:1200]
    instruction = (
        "Translate only the one selected English excerpt into Simplified Chinese. "
        "The surrounding passage is provided for context only, with formulas replaced by …. "
        "Do not repeat other parts of the passage, add explanations, or add notation. "
        "Use these reinforcement learning terms: state = 状态, action = 动作, observation = 观测. "
        "Return one translation for the selected excerpt. Context: " + context
    )
    result = []
    for part in re.split(r"(ZXQ[0-9]{5}QXZ)", unit.value):
        if TOKEN.fullmatch(part) or not re.search(r"[A-Za-z]", part):
            result.append(part)
            continue
        match = re.fullmatch(r"(\s*)(.*?)(\s*)", part, re.S)
        source = match[2]
        if TECHNICAL_ID.fullmatch(source):
            result.append(part)
            continue
        try:
            output = request_fields(options, instruction, {"excerpt": source})[
                "excerpt"
            ]
            validate_text(Text(source), output)
            if (
                TOKEN.search(output)
                or re.search(r"\\[A-Za-z]|[<>{}$…]", output)
                or len(output) > max(12, round(1.2 * len(source)))
            ):
                raise ValueError("the excerpt included material from its context")
        except (OSError, ValueError, TypeError, KeyError, HTTPException):
            # CoMLRL's bounded retries apply to this one small field only.
            output = engine.translate_batch(
                {"s0": source},
                endpoint=options.endpoint,
                model=options.model,
                language="zh-cn",
            )["s0"]
            validate_text(Text(source), output)
        result.extend([match[1], output.strip(), match[3]])
    return "".join(result)


def translate_unit(unit, options, engine):
    if technical_label(unit.value):
        return unit.value
    # One passage per request prevents cross-field leakage on short headings.
    for attempt in range(4):
        try:
            if attempt == 0:
                result = engine._request_batch(
                    {"s0": unit.value},
                    endpoint=options.endpoint,
                    model=options.model,
                    language="zh-cn",
                )["s0"]
            elif attempt == 1 and unit.literals:
                result = retry_markers(unit, options, engine)
            elif attempt < 3:
                result = retry_in_context(unit, options, engine)
            else:
                result = retry_one_segment_at_a_time(unit, options, engine)
            validate_text(unit, result)
            return result
        except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
            if attempt == 3:
                raise ValueError(
                    f"Unable to translate {unit.value[:120]!r}: {exc}"
                ) from exc


def fill_cache(parts, db, options, engine):
    result = {}
    for unit in dict.fromkeys(p for p in parts if isinstance(p, Text)):
        if technical_label(unit.value):
            result[unit] = unit.value
            continue
        key = json.dumps([unit.mode, unit.value, unit.literals], ensure_ascii=False)
        row = db.execute("SELECT target FROM phrases WHERE source=?", (key,)).fetchone()
        if row:
            try:
                validate_text(unit, row[0])
                result[unit] = row[0]
                continue
            except ValueError:
                pass
        if options.cache_only:
            raise ValueError(
                "Missing or invalid cached translation: " + unit.value[:85]
            )
        target = translate_unit(unit, options, engine)
        db.execute("INSERT OR REPLACE INTO phrases VALUES (?, ?)", (key, target))
        db.commit()
        result[unit] = target
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--english-site",
        type=Path,
        required=True,
        help="English Hugo build rooted at /notes/",
    )
    parser.add_argument(
        "--engine-root",
        type=Path,
        default=Path(".translation-engine"),
        help=f"CoMLRL checkout at {ENGINE_REV}",
    )
    parser.add_argument("--content-dir", type=Path, default=Path("notes/content"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path(".cache/blog-translations")
    )
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    options = parser.parse_args()
    if not options.content_dir.is_dir() or not (options.english_site / "index.html").is_file():
        parser.error("Provide the Markdown content directory and a built English blog")
    engine = load_engine(options.engine_root)
    fingerprint = digest(
        b"\0".join(
            [
                b"blog-contextual-passages-v2",
                options.model.encode(),
                Path(engine.__file__).read_bytes(),
                (options.engine_root / "docs/data/languages.json").read_bytes(),
            ]
        )
    )
    options.cache_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    with sqlite3.connect(options.cache_dir / (fingerprint + ".sqlite3")) as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS phrases (source TEXT PRIMARY KEY, target TEXT NOT NULL)"
        )
        for page in sorted(options.content_dir.rglob("*.md")):
            if page.name.endswith(".zh.md") or page.name.startswith("_index."):
                continue
            original = page.read_text(encoding="utf-8")
            front = engine.FRONT.match(original)
            if not front or not re.search(r"^date:[ \t]*\S", front[1], re.M):
                continue
            rel = page.relative_to(options.content_dir)
            english = options.english_site / rel.with_suffix("") / "index.html"
            if not english.is_file():
                # Hugo omits drafts, scheduled articles and expired pages from this build.
                continue
            source = anchor_headings(original, english, engine)
            print(f"Translating {rel} ({options.model})", flush=True)
            parts = list(slots(source, engine, db=db, options=options))
            translations = fill_cache(parts, db, options, engine)
            translation = render(parts, translations)
            if not any(
                unit.mode == "prose" and HAN.search(value)
                for unit, value in translations.items()
            ):
                raise ValueError(f"{rel}: expected Chinese article body")
            outputs[page.with_suffix(".zh.md")] = translation
    # Validate every article before making its Hugo translation available.
    for path, translated in outputs.items():
        if path.exists() and not is_generated(path, engine):
            raise ValueError(f"Refusing to overwrite a hand-authored page: {path}")
    for path, translated in outputs.items():
        with NamedTemporaryFile(
            "w", dir=path.parent, encoding="utf-8", delete=False
        ) as tmp:
            tmp.write(translated)
        Path(tmp.name).replace(path)
    for orphan in options.content_dir.rglob("*.zh.md"):
        if orphan not in outputs and is_generated(orphan, engine):
            orphan.unlink()
    print(f"Prepared {len(outputs)} Simplified Chinese Hugo pages.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        sys.exit(str(exc))
