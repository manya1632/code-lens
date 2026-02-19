"""
CodeLens Inference Pipeline.

Loads a trained model and runs full code review on a file or code string.
Produces a structured ReviewReport with issues, suggestions, and explanations.

Usage:
  python scripts/infer.py --file my_code.py
  python scripts/infer.py --file my_code.js --language javascript
  python scripts/infer.py --checkpoint checkpoints/best_model.pt --file src/main.py
"""

from typing import Optional
import argparse
import torch
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from transformers import AutoTokenizer
from src.model.model import CodeLensModel, BUG_TYPES, COMPLEXITY_CLASSES
from src.model.ast_parser import ASTParser
from src.model.cfg_builder import CFGBuilder
from src.explainability.explainer import CodeExplainer


@dataclass
class Issue:
    bug_type: str
    confidence: float
    severity: str           # critical / high / medium / low
    highlighted_lines: list[int]
    description: str
    suggestion: str


@dataclass
class ReviewReport:
    file: str
    language: str
    overall_severity: str
    severity_score: float
    issues: list[Issue]
    complexity_before: str
    complexity_after: str
    explanation: str
    token_heatmap: list[tuple]    # (token, importance_score)
    score: int                    # 0–100 code quality score


SEVERITY_THRESHOLDS = {
    "critical": 0.8,
    "high": 0.6,
    "medium": 0.4,
    "low": 0.0,
}

ISSUE_DESCRIPTIONS = {
    "security":     ("Potential security vulnerability (injection, unsafe eval, hardcoded credentials, etc.)",
                     "Use parameterized queries, input validation, and secure APIs. Avoid eval() and os.system()."),
    "performance":  ("Inefficient code pattern detected (likely O(n²) or worse complexity)",
                     "Replace nested loops with hash maps or sets. Prefer vectorized operations for bulk data."),
    "memory":       ("Memory management issue (unbounded growth, missing cleanup, potential leak)",
                     "Add size limits to caches. Clear timers/listeners on teardown. Use weak references where appropriate."),
    "logic":        ("Logical error or anti-pattern detected",
                     "Review conditions, boundary cases, and return paths carefully. Avoid bare except clauses."),
    "type_error":   ("Type mismatch or missing type handling",
                     "Add explicit type checks or use type hints. Handle conversion edge cases."),
    "null_deref":   ("Potential null/None dereference",
                     "Add null checks before accessing attributes. Use Optional types and guard clauses."),
    "concurrency":  ("Concurrency / thread-safety issue",
                     "Use locks, thread-safe data structures, or async patterns to protect shared state."),
    "style":        ("Code style or maintainability issue",
                     "Refactor to reduce duplication. Follow language conventions and DRY principles."),
}


class CodeLensInference:
    def __init__(
        self,
        checkpoint_path: str,
        backbone_name: str = "microsoft/graphcodebert-base",
        device: str = "auto",
        explain_method: str = "attention_rollout",
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if device == "auto" else torch.device(device)

        print(f"Loading model from {checkpoint_path} on {self.device}...")
        self.model = CodeLensModel(backbone_name=backbone_name)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(backbone_name)
        self.explainer = CodeExplainer(self.model, self.tokenizer, method=explain_method)

        self._ast_parsers: dict[str, ASTParser] = {}
        self._cfg_builders: dict[str, CFGBuilder] = {}

    def _get_cfg_features(self, code: str, language: str) -> torch.Tensor:
        if language not in self._cfg_builders:
            try:
                self._cfg_builders[language] = CFGBuilder(language)
            except Exception:
                return torch.zeros(1, 7)
        try:
            _, features = self._cfg_builders[language].build(code)
            vec = self._cfg_builders[language].features_to_vector(features)
            return torch.tensor([vec], dtype=torch.float)
        except Exception:
            return torch.zeros(1, 7)

    def review(self, code: str, language: str = "python", file_name: str = "<stdin>") -> ReviewReport:
        """Run full review pipeline on code string."""
        # Tokenize
        encoding = self.tokenizer(
            code,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        cfg_features = self._get_cfg_features(code, language).to(self.device)

        # Forward pass
        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                cfg_features=cfg_features,
            )

        # Explainability
        explanation_data = self.explainer.explain(
            code=code,
            input_ids=input_ids,
            attention_mask=attention_mask,
            cfg_features=cfg_features,
            model_output=output,
        )

        # Build issues list
        severity_score = float(output.severity_score[0])
        overall_severity = next(
            label for label, thresh in SEVERITY_THRESHOLDS.items()
            if severity_score >= thresh
        )

        issues = []
        for i, (prob, bug_type) in enumerate(zip(output.bug_probs[0].tolist(), BUG_TYPES)):
            if prob >= 0.5:
                desc, suggestion = ISSUE_DESCRIPTIONS.get(bug_type, ("Issue detected", "Review carefully."))
                bug_severity = "critical" if prob > 0.85 else "high" if prob > 0.7 else "medium"
                issues.append(Issue(
                    bug_type=bug_type,
                    confidence=round(prob, 3),
                    severity=bug_severity,
                    highlighted_lines=explanation_data["highlighted_lines"],
                    description=desc,
                    suggestion=suggestion,
                ))

        # Complexity predictions
        complexity_before = COMPLEXITY_CLASSES[output.complexity_before_logits[0].argmax().item()]
        complexity_after = COMPLEXITY_CLASSES[output.complexity_after_logits[0].argmax().item()]

        # Quality score (inverse of severity, penalized by number of issues)
        base_score = int((1 - severity_score) * 100)
        penalty = min(len(issues) * 5, 40)
        quality_score = max(0, base_score - penalty)

        return ReviewReport(
            file=file_name,
            language=language,
            overall_severity=overall_severity,
            severity_score=round(severity_score, 3),
            issues=issues,
            complexity_before=complexity_before,
            complexity_after=complexity_after,
            explanation=explanation_data["natural_language"],
            token_heatmap=explanation_data["token_importance"],
            score=quality_score,
        )

    def review_file(self, file_path: str, language: Optional[str] = None) -> ReviewReport:
        path = Path(file_path)
        code = path.read_text(encoding="utf-8")
        lang = language or self._detect_language(path.suffix)
        return self.review(code, language=lang, file_name=str(path))

    def _detect_language(self, suffix: str) -> str:
        return {
            ".py": "python", ".js": "javascript", ".ts": "javascript",
            ".java": "java", ".cpp": "cpp", ".go": "go", ".rb": "ruby",
        }.get(suffix, "python")

    def print_report(self, report: ReviewReport):
        print("\n" + "=" * 60)
        print(f"  CodeLens Review: {report.file}")
        print("=" * 60)
        print(f"  Quality Score : {report.score}/100")
        print(f"  Severity      : {report.overall_severity.upper()} ({report.severity_score:.0%})")
        print(f"  Complexity    : {report.complexity_before} → {report.complexity_after}")
        print(f"  Issues Found  : {len(report.issues)}")
        print()

        for i, issue in enumerate(report.issues, 1):
            print(f"  [{i}] {issue.bug_type.upper()} ({issue.severity}) — confidence {issue.confidence:.0%}")
            print(f"      Lines: {issue.highlighted_lines}")
            print(f"      {issue.description}")
            print(f"      Fix: {issue.suggestion}")
            print()

        print(f"  Explanation:\n  {report.explanation}")
        print()
        print(f"  Top influential tokens:")
        for tok, score in report.token_heatmap[:5]:
            print(f"    {tok:20s} → {score:.4f}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Source file to review")
    parser.add_argument("--language", default=None)
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--backbone", default="microsoft/graphcodebert-base")
    parser.add_argument("--output_json", default=None, help="Save report to JSON file")
    parser.add_argument("--explain_method", default="attention_rollout",
                        choices=["attention_rollout", "integrated_gradients"])
    args = parser.parse_args()

    pipeline = CodeLensInference(
        checkpoint_path=args.checkpoint,
        backbone_name=args.backbone,
        explain_method=args.explain_method,
    )

    report = pipeline.review_file(args.file, args.language)
    pipeline.print_report(report)

    if args.output_json:
        with open(args.output_json, "w") as f:
            # Convert dataclass to dict, handle nested dataclasses
            report_dict = asdict(report)
            json.dump(report_dict, f, indent=2)
        print(f"\nReport saved to {args.output_json}")


if __name__ == "__main__":
    main()
