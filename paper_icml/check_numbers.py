"""Flag any number in main.tex that appears neither in the original manuscript, results/,
nor the archived supplement (supplements/transfer-comparison-2026-11-20/).

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
SUPP = ROOT / "supplements" / "transfer-comparison-2026-11-20"
sources = list((ROOT / "results").rglob("*.csv")) + list((ROOT / "results").rglob("*.md"))
for ext in ("*.csv", "*.md", "*.txt", "*.json"):
    sources += list(SUPP.rglob(ext))
for path in sources:
    pool += path.read_text(encoding="utf-8", errors="ignore")
numbers = set(re.findall(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?", text))
unmatched = sorted((n for n in numbers if n not in pool), key=lambda n: float(n.replace(",", "")))
print("Numbers not found verbatim in paper_draft/main.tex, results/ or the supplement:", unmatched or "none")
