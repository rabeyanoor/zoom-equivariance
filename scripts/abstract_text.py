"""Plain-text abstract for the submission form: sec/0_abstract.tex with every number macro filled in.

    python scripts/abstract_text.py        # -> paper/gen/abstract.txt
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", (ROOT / "paper/gen/numbers.tex").read_text()))
text = (ROOT / "paper/sec/0_abstract.tex").read_text()
text = re.sub(r"\\(begin|end)\{abstract\}", "", text)
text = re.sub(r"\\([A-Za-z]+)(\{\})?", lambda m: macros.get(m.group(1), m.group(0)), text)
text = text.replace("--", "–").replace("``", '"').replace("''", '"')


def math(m):
    s = m.group(1)
    for a, b in ((r"\alpha", "α"), (r"\beta", "β"), (r"\approx", "≈")):
        s = s.replace(a, b)
    return s.replace("s^{-α}", "s^(−α)").replace("s^{β}", "s^β")


text = " ".join(re.sub(r"\$([^$]*)\$", math, text).split())
text = re.sub(r"(?<![\w.])-(?=\d)", "−", text)  # minus signs of the numbers
assert "\\" not in text and "$" not in text, text
out = ROOT / "paper/gen/abstract.txt"
out.write_text(text + "\n")
print(text)
print(f"{len(text.split())} words, {len(text)} characters -> {out}")
