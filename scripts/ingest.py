"""Convert PDFs in data/papers/ into clean .txt files in data/corpus/ and log provenance.
Usage: python scripts/ingest.py [--src data/papers] [--dst data/corpus]
Keep a license note per paper in data/papers/SOURCES.csv: file,title,url,license"""
import argparse, csv, re, subprocess
from pathlib import Path

def clean(t: str) -> str:
    t = re.sub(r"-\n(\w)", r"\1", t)           # de-hyphenate line breaks
    t = re.sub(r"\n(?=[a-z])", " ", t)         # join wrapped lines
    t = re.sub(r"\[\d+(?:[,\u2013-]\s*\d+)*\]", "", t)  # drop numeric citations
    t = re.sub(r"[ \t]+", " ", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/papers"); ap.add_argument("--dst", default="data/corpus")
    a = ap.parse_args()
    dst = Path(a.dst); dst.mkdir(parents=True, exist_ok=True)
    log = []
    for pdf in sorted(Path(a.src).glob("*.pdf")):
        raw = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True).stdout
        out = dst / f"{pdf.stem}.txt"
        out.write_text(clean(raw), encoding="utf-8")
        log.append((pdf.name, len(raw.split())))
        print(f"{pdf.name}: {len(raw.split())} words -> {out}")
    with open(dst / "_ingest_log.csv", "w", newline="") as f:
        csv.writer(f).writerows([("file", "words")] + log)

if __name__ == "__main__":
    main()
