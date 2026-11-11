"""Offline regression tests. Set COMLRL_ENGINE to the pinned CoMLRL checkout."""

import os
import re
from pathlib import Path
from types import SimpleNamespace
import sqlite3
import unittest
from unittest.mock import patch

from translate_blog import (
    Text,
    fill_cache,
    load_engine,
    render,
    slots,
    validate_text,
    translate_unit,
    restore,
)


class BlogTranslationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = load_engine(
            Path(os.environ.get("COMLRL_ENGINE", ".translation-engine"))
        )

    def test_translate_only_prose(self):
        protected = [
            "({{< katex >}}\\theta{{< /katex >}})",
            r"\(Q^*(s,a)\)",
            "$x_1$",
            "{#policy}",
            "Ouyang et al., 2022",
            "Shi et al. (2025, revised 2026)",
            "Sutton and Barto 2018, Sec. 6.7",
            "Lai et. al, 2024",
            '<a href="#ref:x">',
            "`agent.step()`",
            "https://example.org/p_(1).pdf",
            "```python\nprint('English code')\n```",
            "{{< references >}}\n<li>A. Author, English paper title.</li>\n{{< /references >}}",
        ]
        sample = (
            '---\ntitle: "Testing an article"\ndescription: "An English abstract."\n'
            "date: 2025-02-01\ntags: [RL, Math]\n---\n"
            "## Heading {#policy}\n"
            "Let ({{< katex >}}\\theta{{< /katex >}}), \\(Q^*(s,a)\\) and $x_1$ be variables.\n"
            "[Ouyang et al., 2022](https://example.org/p_(1).pdf) and "
            "Shi et al. (2025, revised 2026); (Sutton and Barto 2018, Sec. 6.7); "
            "(Lai et. al, 2024).\n"
            '<a href="#ref:x">see equation</a> or use `agent.step()`.\n'
            "| Term | Meaning |\n| :--- | ---: |\n"
            '{{% details "Proof of theorem" %}}\nMore text.\n{{% /details %}}\n'
            + protected[-2]
            + "\n"
            + protected[-1]
            + "\n"
        )
        parts = list(slots(sample, self.engine))
        translations = {
            p: re.sub(
                r"ZXQ[0-9]{5}QXZ|[A-Za-z]+",
                lambda m: m[0] if m[0].startswith("ZXQ") else "中文",
                p.value,
            )
            for p in parts
            if isinstance(p, Text)
        }
        target = render(parts, translations)
        for span in protected:
            self.assertIn(span, target)
        self.assertIn("date: 2025-02-01\ntags: [RL, Math]", target)
        self.assertIn("| 中文 | 中文 |\n| :--- | ---: |", target)
        self.assertIn('{{% details "中文 中文 中文" %}}', target)
        full_line = next(
            p for p in parts if isinstance(p, Text) and p.value.startswith("Let ")
        )
        self.assertIn("be variables.", full_line.value)
        self.assertIn("$x_1$", "".join(full_line.literals))
        self.assertNotIn("An English abstract", target)

    def test_whole_sentence_and_link_label(self):
        db = sqlite3.connect(":memory:")
        self.addCleanup(db.close)
        db.execute(
            "CREATE TABLE phrases (source TEXT PRIMARY KEY, target TEXT NOT NULL)"
        )
        options = SimpleNamespace(model="test", endpoint="unused", cache_only=False)
        source = "---\ntitle: DPO\ndate: 2026-01-01\n---\nThe [reward model](https://example.org) scores $x$ and $y$.\n"
        with patch.object(
            self.engine, "_request_batch", return_value={"s0": "奖励模型"}
        ):
            parts = list(slots(source, self.engine, db=db, options=options))
        unit = next(p for p in parts if isinstance(p, Text) and p.mode == "prose")
        self.assertEqual(unit.literals[0], "[奖励模型](https://example.org)")
        self.assertIn("The ZXQ00000QXZ scores ZXQ00001QXZ and ZXQ00002QXZ", unit.value)
        with self.assertRaisesRegex(ValueError, "placeholder"):
            validate_text(unit, "未能保留公式")
        with self.assertRaisesRegex(ValueError, "markup"):
            validate_text(unit, "又附上了 https://example.org：" + unit.value)
        validate_text(Text("DPO", "yaml"), "DPO")
        for label in ("GPT-4", "Llama 2", "PettingZoo", "Q-learning", "MMDP (Cen-MMDP)"):
            with self.subTest(label=label):
                self.assertEqual(translate_unit(Text(label, "link"), options, None), label)
        with self.assertRaisesRegex(ValueError, "Chinese"):
            validate_text(
                Text("Learning from Preferences", "yaml"), "Learning from Preferences"
            )

    def test_bold_label_stays_attached_to_its_translation(self):
        db = sqlite3.connect(":memory:")
        self.addCleanup(db.close)
        db.execute("CREATE TABLE phrases (source TEXT PRIMARY KEY, target TEXT NOT NULL)")
        options = SimpleNamespace(model="test", endpoint="unused", cache_only=False)
        source = (
            "---\ntitle: DPO\ndate: 2026-01-01\n---\n"
            "The **full gradient** includes the target gradient.\n"
            "Keep `**literal**` and $x^{**}$ intact.\n"
        )
        with patch.object(self.engine, "_request_batch", return_value={"s0": "全梯度"}):
            parts = list(slots(source, self.engine, db=db, options=options))
        body = [p for p in parts if isinstance(p, Text) and p.mode == "prose"]
        self.assertEqual(
            restore(body[0], "ZXQ00000QXZ包含目标梯度。"),
            "<strong>全梯度</strong>包含目标梯度。",
        )
        self.assertIn("`**literal**`", body[1].literals)
        self.assertIn("$x^{**}$", body[1].literals)

    def test_long_math_line_recovers_each_gap_once(self):
        unit = Text(
            "ZXQ00000QXZ, ZXQ00001QXZ: ZXQ00002QXZ given the actions ZXQ00003QXZ;",
            "prose",
            ("$o$", "$s$", "$p$", "$a$"),
        )
        options = SimpleNamespace(endpoint="unused", model="test")
        with patch.object(
            self.engine, "_request_batch", side_effect=ValueError("broken markers")
        ), patch(
            "translate_blog.retry_in_context", side_effect=ValueError("truncated JSON")
        ), patch(
            "translate_blog.request_fields", side_effect=ValueError("truncated JSON")
        ) as excerpts, patch.object(
            self.engine, "translate_batch", return_value={"s0": "给定动作"}
        ) as isolated:
            target = translate_unit(unit, options, self.engine)
        self.assertEqual(restore(unit, target), "$o$, $s$: $p$ 给定动作 $a$;")
        self.assertEqual(excerpts.call_count, 1)
        self.assertEqual(isolated.call_args.args[0], {"s0": "given the actions"})

    def test_cached_run_and_english_echo(self):
        options = SimpleNamespace(endpoint="unused", model="test", cache_only=False)
        db = sqlite3.connect(":memory:")
        self.addCleanup(db.close)
        db.execute(
            "CREATE TABLE phrases (source TEXT PRIMARY KEY, target TEXT NOT NULL)"
        )
        fields = [Text("An ordinary sentence needs translation.")]

        def echo(values, **kwargs):
            return dict(values)

        with patch.object(self.engine, "_request_batch", side_effect=echo), patch(
            "translate_blog.retry_in_context", return_value=fields[0].value
        ), patch(
            "translate_blog.retry_one_segment_at_a_time", return_value=fields[0].value
        ):
            with self.assertRaisesRegex(ValueError, "no Chinese translation"):
                fill_cache(fields, db, options, self.engine)
        with patch.object(
            self.engine, "_request_batch", return_value={"s0": "这句话需要翻译。"}
        ):
            translated = fill_cache(fields, db, options, self.engine)
        options.cache_only = True
        with patch.object(self.engine, "_request_batch", side_effect=AssertionError):
            self.assertEqual(fill_cache(fields, db, options, self.engine), translated)
            with self.assertRaisesRegex(ValueError, "Missing"):
                fill_cache(
                    [Text("Another English sentence.")], db, options, self.engine
                )


if __name__ == "__main__":
    unittest.main()
