import json, re
from pathlib import Path

BLOCK = re.compile(r"<<<<\s*SEARCH\n(.*?)\n====\s*REPLACE\n(.*?)\n>>>>\s*END", re.S)

def apply_sr(before_text: str, sr_text: str) -> str:
    out = before_text
    blocks = BLOCK.findall(sr_text)
    if not blocks:
        raise ValueError("No SEARCH/REPLACE blocks found in assistant output.")
    for search, replace in blocks:
        idx = out.find(search)
        if idx < 0:
            raise ValueError("SEARCH block not found in current text.")
        out = out[:idx] + replace + out[idx+len(search):]
    return out

def main(out_dir: str, jsonl_path: str):
    out_dir = Path(out_dir).resolve()
    jsonl_path = Path(jsonl_path).resolve()

    bad = 0
    n = 0
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        n += 1
        rec = json.loads(line)
        msgs = rec["messages"]
        user = next(m["content"] for m in msgs if m["role"] == "user")
        assistant = next(m["content"] for m in msgs if m["role"] == "assistant")

        m = re.search(r"File:\s*(.+)\n", user)
        if not m:
            print("Skip: cannot find File: ... in prompt")
            bad += 1
            continue
        rel = m.group(1).strip()

        bf = out_dir / "before" / rel
        af = out_dir / "after" / rel
        if not bf.exists() or not af.exists():
            print(f"Missing before/after file for: {rel}")
            bad += 1
            continue

        before_text = bf.read_text(encoding="utf-8")
        after_text  = af.read_text(encoding="utf-8")

        try:
            pred = apply_sr(before_text, assistant)
        except Exception as e:
            print(f"[FAIL] {rel}: {e}")
            bad += 1
            continue

        if pred != after_text:
            print(f"[FAIL] {rel}: transformed != after")
            bad += 1
        else:
            print(f"[OK]   {rel}")

    print(f"\nChecked {n} examples, bad={bad}")

if __name__ == "__main__":
    # 用法：python validate_sr.py ../_sft_out_jpetstore6 ../_sft_out_jpetstore6/sr_sft.jsonl
    import sys
    main(sys.argv[1], sys.argv[2])
