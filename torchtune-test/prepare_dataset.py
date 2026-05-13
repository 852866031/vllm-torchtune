"""Convert one-example-per-line Alpaca-style .txt into JSON for torchtune."""
import argparse, json, re, sys
from pathlib import Path

PAT = re.compile(
    r"### Instruction:\s*(?P<instruction>.*?)"
    r"(?:### Input:\s*(?P<input>.*?))?"
    r"### Response:\s*(?P<output>.*)$",
    re.DOTALL,
)


def parse_line(line: str):
    line = line.strip()
    if not line:
        return None
    m = PAT.search(line)
    if not m:
        return None
    return {
        "instruction": (m.group("instruction") or "").strip(),
        "input": (m.group("input") or "").strip(),
        "output": (m.group("output") or "").strip(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, help="input .txt")
    ap.add_argument("--out", dest="dst", required=True, help="output .json")
    args = ap.parse_args()

    rows, skipped = [], 0
    for raw in Path(args.src).read_text(encoding="utf-8").splitlines():
        row = parse_line(raw)
        if row is None:
            skipped += 1
            continue
        rows.append(row)

    Path(args.dst).write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"wrote {len(rows)} rows to {args.dst} (skipped {skipped})", file=sys.stderr)


if __name__ == "__main__":
    main()
