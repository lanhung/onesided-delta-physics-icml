"""Flag any number in main.tex that appears neither in the original manuscript nor in results/.

Generated tables (tables/*.tex) are excluded because make_icml_tables.py builds them
directly from the CSVs. Numbers flagged here must be checked by hand; the expected
output lists only illustrative values derived in the text (e.g. the per-step damping
factor 0.998 = exp(-1e-3 * (2*pi)^2 * 0.05)).
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
text = "\n".join(line.split("%%")[0] for line in (HERE / "main.tex").read_text(encoding="utf-8").splitlines())
pool = (ROOT / "paper_draft" / "main.tex").read_text(encoding="utf-8")
for path in list((ROOT / "results").rglob("*.csv")) + list((ROOT / "results").rglob("*.md")):
    pool += path.read_text(encoding="utf-8")
numbers = set(re.findall(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?", text))
unmatched = sorted((n for n in numbers if n not in pool), key=lambda n: float(n.replace(",", "")))
print("Numbers not found verbatim in paper_draft/main.tex or results/:", unmatched or "none")
