#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import difflib
import json
from pathlib import Path

def build_hunks(a_lines, b_lines, context=2):
    """
    用 SequenceMatcher 得到 opcodes，再把每个 change 扩展 context 行，
    同时合并重叠 hunks，返回 (a_start,a_end,b_start,b_end) 列表。
    """
    sm = difflib.SequenceMatcher(a=a_lines, b=b_lines)
    opcodes = sm.get_opcodes()

    raw = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue
        a0 = max(0, i1 - context)
        a1 = min(len(a_lines), i2 + context)
        b0 = max(0, j1 - context)
        b1 = min(len(b_lines), j2 + context)
        raw.append((a0, a1, b0, b1))

    if not raw:
        return []

    raw.sort()
    merged = [raw[0]]
    for a0, a1, b0, b1 in raw[1:]:
        pa0, pa1, pb0, pb1 = merged[-1]
        # 有重叠/相邻就合并
        if a0 <= pa1 and b0 <= pb1:
            merged[-1] = (min(pa0, a0), max(pa1, a1), min(pb0, b0), max(pb1, b1))
        else:
            merged.append((a0, a1, b0, b1))
    return merged

def to_block(relpath, search_text, replace_text):
    return (
        f"### FILE: {relpath}\n"
        f"<<<< SEARCH\n{search_text}\n"
        f"==== REPLACE\n{replace_text}\n"
        f">>>> END\n"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="例如 ../_sft_out_jpetstore6")
    ap.add_argument("--recipe", required=True, help="例如 org.openrewrite.staticanalysis.CommonStaticAnalysis")
    ap.add_argument("--context", type=int, default=2)
    ap.add_argument("--files", nargs="+", required=True, help="例如 src/main/.../Cart.java ...")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    before_root = out_dir / "before"   
    after_root = out_dir / "after"

    blocks_out = out_dir / "sr_blocks.txt"
    jsonl_out = out_dir / "sft.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_blocks = []
    jsonl_rows = []

    for rel in args.files:
        rel = rel.strip()
        bf = before_root / rel
        af = after_root / rel

        if not bf.exists():
            raise FileNotFoundError(f"before 文件不存在: {bf}")
        if not af.exists():
            raise FileNotFoundError(f"after 文件不存在: {af}")

        a_text = bf.read_text(encoding="utf-8", errors="replace")
        b_text = af.read_text(encoding="utf-8", errors="replace")

        a_lines = a_text.splitlines()
        b_lines = b_text.splitlines()

        hunks = build_hunks(a_lines, b_lines, context=args.context)
        if not hunks:
            continue

        file_blocks = []
        for a0, a1, b0, b1 in hunks:
            search = "\n".join(a_lines[a0:a1])
            repl = "\n".join(b_lines[b0:b1])
            file_blocks.append(to_block(rel, search, repl))

        label = "\n".join(file_blocks).rstrip() + "\n"
        all_blocks.append(label)

        user_prompt = (
            f"Task: Refactor Java code using OpenRewrite recipe.\n"
            f"Recipe: {args.recipe}\n"
            f"Output format: Search & Replace blocks ONLY.\n"
            f"File: {rel}\n\n"
            f"--- BEFORE FILE CONTENT ---\n{a_text}\n"
        )

        jsonl_rows.append({
            "messages": [
                {"role": "system", "content": "You are a code refactoring assistant. Output ONLY Search & Replace blocks."},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": label}
            ]
        })

    blocks_out.write_text("\n".join(all_blocks).rstrip() + "\n", encoding="utf-8")
    with jsonl_out.open("w", encoding="utf-8") as f:
        for row in jsonl_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[OK] blocks: {blocks_out}")
    print(f"[OK] jsonl : {jsonl_out}")
    print(f"[OK] samples: {len(jsonl_rows)}")

if __name__ == "__main__":
    main()
