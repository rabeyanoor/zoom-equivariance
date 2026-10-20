"""Collect every run log into logs/ as plain text, so the history of each experiment is kept.

    python scripts/save_logs.py

- Kaggle kernel logs (JSON lines with time stamps) from outputs/kaggle/*/  ->  logs/kaggle/<kernel>.log
- local logs (push queue, downloads) are written straight to logs/local/ by the scripts (kept on the laptop, not in git)
A log that already exists is never overwritten: a new version gets a numbered name.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"


def free_name(folder, stem, ext):
    p = folder / f"{stem}{ext}"
    k = 2
    while p.exists():
        p = folder / f"{stem}__v{k}{ext}"
        k += 1
    return p


def same_as_existing(folder, stem, text):
    return any(f.read_text() == text for f in folder.glob(f"{stem}*.log"))


def kaggle_to_text(path):
    lines = []
    for entry in json.loads(path.read_text()):
        stream = "ERR" if entry.get("stream_name") == "stderr" else "OUT"
        for line in entry.get("data", "").splitlines():
            if line.strip():
                lines.append(f"[{entry.get('time', 0):8.1f}s {stream}] {line}")
    return "\n".join(lines) + "\n"


def main():
    (LOGS / "kaggle").mkdir(parents=True, exist_ok=True)
    (LOGS / "local").mkdir(parents=True, exist_ok=True)
    n = 0
    for log in sorted(ROOT.glob("outputs/kaggle/*/*.log")):
        try:
            text = kaggle_to_text(log)
        except json.JSONDecodeError:
            continue  # not a kernel log (e.g. a LaTeX log in the kernel's output)
        stem = log.stem
        if same_as_existing(LOGS / "kaggle", stem, text):
            continue
        free_name(LOGS / "kaggle", stem, ".log").write_text(text)
        n += 1
    print(n, "logs saved to", LOGS)


main()
