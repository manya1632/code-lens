"""
Preprocess raw datasets into unified JSONL format for CodeLens training.

Supported sources:
  - Bugs2Fix (Tufano et al.) — pairs of buggy/fixed Java methods
  - CodeSearchNet — Python, JS, Java, Go, PHP, Ruby functions
  - Custom CSV/JSON user datasets

Output format (one JSON per line):
{
  "code": "...",
  "language": "python",
  "bugs": ["performance"],
  "severity": 0.7,
  "complexity_before": "O(n²)",
  "complexity_after": "O(n)",
  "fixed_code": "..."
}
"""
import json
import random
import argparse
import re
from pathlib import Path
from tqdm import tqdm


BUG_HEURISTICS = {
    "security": [
    r"execute\s*\(.*%s",         
    r"SELECT.*%s",               
    r"%\s*\w+",                  
    r"eval\s*\(",
    r"pickle\.loads",
    r"os\.system",
    r"password.*=.*['\"]",
    ],

    "performance": [
        r"for\s+\w+\s+in.*:\s*\n.*for\s+\w+\s+in",   
        r"\.append\(.*\).*in.*range",                  
        r"\+\s*=\s*str\(",                             
    ],
    "memory": [
        r"while\s+True.*cache\[",     
        r"setInterval|setInterval",   
        r"addEventListener.*function",
    ],
    "logic": [
        r"if\s+\w+\s*==\s*True",     
        r"except\s*:\s*pass",         
        r"range\(len\(",               
    ],
}


def detect_bug_types(code: str) -> list[str]:
    """Heuristically detect bug types from code patterns."""
    detected = []
    for bug_type, patterns in BUG_HEURISTICS.items():
        for pattern in patterns:
            if re.search(pattern, code, re.IGNORECASE | re.MULTILINE):
                detected.append(bug_type)
                break
    return detected or ["style"]


def estimate_complexity(code: str) -> str:
    """
    Rough heuristic complexity estimation.
    Replace with ML-based prediction after model training.
    """
    loop_depth = 0
    max_depth = 0
    lines = code.split("\n")

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith(("for ", "while ")):
            loop_depth += 1
            max_depth = max(max_depth, loop_depth)
        elif stripped.startswith(("return", "def ", "class ")):
            loop_depth = max(0, loop_depth - 1)

    if max_depth == 0:
        return "O(1)"
    elif max_depth == 1:
        # Check for sorting calls
        if any(kw in code for kw in ["sorted(", ".sort(", "heapq"]):
            return "O(n log n)"
        return "O(n)"
    elif max_depth == 2:
        return "O(n²)"
    elif max_depth >= 3:
        return "O(n³)"
    return "O(n)"


def estimate_severity(bugs: list[str]) -> float:
    """Map bug types to severity scores."""
    severity_map = {
        "security": 1.0,
        "memory": 0.8,
        "performance": 0.7,
        "logic": 0.6,
        "type_error": 0.5,
        "null_deref": 0.5,
        "concurrency": 0.9,
        "style": 0.2,
    }
    if not bugs:
        return 0.1
    return max(severity_map.get(b, 0.3) for b in bugs)


def process_bugs2fix(input_dir: Path) -> list[dict]:
    """
    Bugs2Fix dataset structure:
      input_dir/
        train.buggy-fixed.buggy   — one method per line
        train.buggy-fixed.fixed
    """
    samples = []
    for split in ["train", "val", "test"]:
        buggy_file = input_dir / f"{split}.buggy-fixed.buggy"
        fixed_file = input_dir / f"{split}.buggy-fixed.fixed"
        if not buggy_file.exists():
            continue

        with open(buggy_file) as bf, open(fixed_file) as ff:
            for buggy, fixed in tqdm(zip(bf, ff), desc=f"Bugs2Fix {split}"):
                buggy, fixed = buggy.strip(), fixed.strip()
                if not buggy or len(buggy) < 20:
                    continue
                bugs = detect_bug_types(buggy)
                complexity = estimate_complexity(buggy)
                fixed_complexity = estimate_complexity(fixed)
                samples.append({
                    "code": buggy,
                    "fixed_code": fixed,
                    "language": "java",
                    "bugs": bugs,
                    "severity": estimate_severity(bugs),
                    "complexity_before": complexity,
                    "complexity_after": fixed_complexity,
                    "source": "bugs2fix",
                })
    return samples


def process_codesearchnet(input_dir: Path, language: str) -> list[dict]:
    """
    CodeSearchNet format: one JSONL per language split.
    We treat clean code as having no bugs (severity=0, style only).
    """
    samples = []
    for split_file in input_dir.glob("*.jsonl"):
        with open(split_file) as f:
            for line in tqdm(f, desc=f"CSN {language} {split_file.name}"):
                obj = json.loads(line.strip())
                code = obj.get("code", "")
                if len(code) < 30:
                    continue
                bugs = detect_bug_types(code)
                samples.append({
                    "code": code,
                    "fixed_code": "",
                    "language": language,
                    "bugs": bugs,
                    "severity": estimate_severity(bugs),
                    "complexity_before": estimate_complexity(code),
                    "complexity_after": estimate_complexity(code),
                    "source": "codesearchnet",
                })
    return samples


def split_and_save(samples: list[dict], output_dir: Path, ratios: tuple):
    output_dir.mkdir(parents=True, exist_ok=True)
    random.shuffle(samples)
    n = len(samples)
    train_end = int(n * ratios[0])
    val_end = train_end + int(n * ratios[1])

    splits = {
        "train": samples[:train_end],
        "val": samples[train_end:val_end],
        "test": samples[val_end:],
    }
    for split_name, split_data in splits.items():
        out_path = output_dir / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for sample in split_data:
                f.write(json.dumps(sample) + "\n")
        print(f"Wrote {len(split_data)} samples to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["bugs2fix", "codesearchnet", "custom"], required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--language", default="python", help="For CodeSearchNet")
    parser.add_argument("--split", nargs=3, type=float, default=[0.8, 0.1, 0.1])
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if args.source == "bugs2fix":
        samples = process_bugs2fix(input_dir)
    elif args.source == "codesearchnet":
        samples = process_codesearchnet(input_dir, args.language)
    else:
        raise NotImplementedError("Custom format: load your own JSONL")

    print(f"Total samples: {len(samples)}")
    split_and_save(samples, output_dir, tuple(args.split))


if __name__ == "__main__":
    main()
