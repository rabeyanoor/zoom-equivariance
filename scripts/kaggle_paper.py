"""Compile the paper on a Kaggle CPU kernel (the laptop has no LaTeX).

The paper sources go up as a private Kaggle dataset (figures make them too big to embed in the
kernel), and the kernel reads them from /kaggle/input.

    python scripts/kaggle_paper.py rabeya100x
    kaggle datasets version -p kaggle/build/paper_data -m update --dir-mode zip   (first time: datasets create)
    kaggle kernels push -p kaggle/build/paper
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
user = sys.argv[1]
paper = ROOT / "paper"
data = ROOT / "kaggle" / "build" / "paper_data"
if data.exists():
    shutil.rmtree(data)
for p in sorted(paper.rglob("*")):
    built = {"main.pdf", "supp.pdf", "main_named.pdf"}
    if p.is_file() and "kit" not in p.parts and p.suffix in {".tex", ".sty", ".bst", ".bib", ".pdf"} and p.name not in built:
        dst = data / p.relative_to(paper)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, dst)
(data / "dataset-metadata.json").write_text(json.dumps({
    "title": "zoom-eq-paper", "id": f"{user}/zoom-eq-paper", "licenses": [{"name": "CC0-1.0"}]}, indent=2))
body = '''
import shutil, subprocess
from pathlib import Path
src = Path("/tmp/paper")
found = [p.parent for p in Path("/kaggle/input").rglob("main.tex")]
shutil.copytree(found[0], src)
def sh(c):
    print("+", c, flush=True)
    return subprocess.run(c, shell=True).returncode
sh("apt-get -qq update && DEBIAN_FRONTEND=noninteractive apt-get -qq install -y --no-install-recommends "
   "texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended texlive-science latexmk > /dev/null")
rc = sh("cd /tmp/paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex > build.log 2>&1")
# the supplement reads main.aux for its cross-references, so it is built second
rc_supp = sh("cd /tmp/paper && latexmk -pdf -interaction=nonstopmode -halt-on-error supp.tex > build_supp.log 2>&1")
sh("cd /tmp/paper && grep -nE '^!|Warning.*undefined|Overfull' build_supp.log | head -40")
sh("cd /tmp/paper && grep -nE '^!|Warning.*undefined|Overfull|Citation .* undefined|Reference .* undefined' build.log | head -80")
# the same paper with the author shown (main_named.tex), for arXiv or the camera-ready copy
rc_named = sh("cd /tmp/paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main_named.tex > build_named.log 2>&1")
sh("cp /tmp/paper/main.pdf /tmp/paper/build.log /tmp/paper/main.log /kaggle/working/ 2>/dev/null")
sh("cp /tmp/paper/main_named.pdf /kaggle/working/ 2>/dev/null")
sh("cp /tmp/paper/supp.pdf /tmp/paper/build_supp.log /tmp/paper/supp.log /kaggle/working/ 2>/dev/null")
print("latexmk exit code", rc, "named", rc_named, "supplement", rc_supp, flush=True)
'''
out = ROOT / "kaggle" / "build" / "paper"
out.mkdir(parents=True, exist_ok=True)
(out / "kernel.py").write_text(body)
(out / "kernel-metadata.json").write_text(json.dumps({
    "id": f"{user}/zoom-equivariance-paper", "title": "zoom-equivariance-paper", "code_file": "kernel.py",
    "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False,
    "enable_internet": True, "dataset_sources": [f"{user}/zoom-eq-paper"]}, indent=2))
print("built", out, "and", data)
