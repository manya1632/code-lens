"""
Evaluate a trained CodeLens model on the test set.
Produces per-class metrics, confusion matrices, and a full report.

Usage:
  python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --test_data data/processed/test.jsonl \
    --output_dir evaluation/
"""
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix, mean_absolute_error,
)
from transformers import AutoTokenizer
from src.model.model import CodeLensModel, BUG_TYPES, COMPLEXITY_CLASSES
from src.data.dataset import CodeReviewDataset, collate_fn
from torch.utils.data import DataLoader


def evaluate(
    checkpoint_path: str,
    test_data_path: str,
    backbone_name: str = "microsoft/graphcodebert-base",
    batch_size: int = 16,
    output_dir: str = "evaluation/",
    threshold: float = 0.5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = CodeLensModel(backbone_name=backbone_name)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # Load test data
    dataset = CodeReviewDataset(test_data_path, tokenizer_name=backbone_name)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=2)

    all_bug_probs, all_bug_labels = [], []
    all_sev_preds, all_sev_labels = [], []
    all_cpx_before_preds, all_cpx_before_labels = [], []
    all_cpx_after_preds, all_cpx_after_labels = [], []

    print("Running evaluation...")
    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            cfg_features = batch["cfg_features"].to(device)

            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                cfg_features=cfg_features,
            )

            all_bug_probs.extend(output.bug_probs.cpu().numpy().tolist())
            all_bug_labels.extend(batch["labels_bug"].numpy().tolist())
            all_sev_preds.extend(output.severity_score.cpu().numpy().tolist())
            all_sev_labels.extend(batch["labels_severity"].numpy().tolist())
            all_cpx_before_preds.extend(output.complexity_before_logits.argmax(-1).cpu().numpy().tolist())
            all_cpx_before_labels.extend(batch["labels_complexity_before"].numpy().tolist())
            all_cpx_after_preds.extend(output.complexity_after_logits.argmax(-1).cpu().numpy().tolist())
            all_cpx_after_labels.extend(batch["labels_complexity_after"].numpy().tolist())

    bug_probs = np.array(all_bug_probs)
    bug_labels = np.array(all_bug_labels)
    bug_preds = (bug_probs >= threshold).astype(int)

    # ── Bug classification report ──────────────────────────────────────────
    print("\n=== Bug Classification ===")
    report = classification_report(
        bug_labels, bug_preds,
        target_names=BUG_TYPES,
        zero_division=0,
    )
    print(report)
    (output_dir / "bug_classification_report.txt").write_text(report)

    # Per-class AUC
    auc_scores = {}
    ap_scores = {}
    for i, bug_type in enumerate(BUG_TYPES):
        if bug_labels[:, i].sum() == 0:
            continue
        auc_scores[bug_type] = roc_auc_score(bug_labels[:, i], bug_probs[:, i])
        ap_scores[bug_type] = average_precision_score(bug_labels[:, i], bug_probs[:, i])

    print("\nPer-class AUC-ROC:")
    for bt, auc in auc_scores.items():
        print(f"  {bt:15s}: {auc:.4f}  (AP={ap_scores[bt]:.4f})")

    # ── Severity regression ─────────────────────────────────────────────────
    sev_mae = mean_absolute_error(all_sev_labels, all_sev_preds)
    sev_preds_arr = np.array(all_sev_preds)
    sev_labels_arr = np.array(all_sev_labels)
    sev_rmse = float(np.sqrt(np.mean((sev_preds_arr - sev_labels_arr) ** 2)))
    print(f"\n=== Severity Regression ===")
    print(f"  MAE  : {sev_mae:.4f}")
    print(f"  RMSE : {sev_rmse:.4f}")

    # ── Complexity classification ───────────────────────────────────────────
    cpx_before_acc = np.mean(np.array(all_cpx_before_preds) == np.array(all_cpx_before_labels))
    cpx_after_acc = np.mean(np.array(all_cpx_after_preds) == np.array(all_cpx_after_labels))
    print(f"\n=== Complexity Prediction ===")
    print(f"  Before Accuracy : {cpx_before_acc:.4f}")
    print(f"  After Accuracy  : {cpx_after_acc:.4f}")

    # ── Save summary JSON ───────────────────────────────────────────────────
    summary = {
        "bug_detection": {
            "auc_per_class": auc_scores,
            "ap_per_class": ap_scores,
            "mean_auc": float(np.mean(list(auc_scores.values()))),
            "mean_ap": float(np.mean(list(ap_scores.values()))),
        },
        "severity": {
            "mae": sev_mae,
            "rmse": sev_rmse,
        },
        "complexity": {
            "before_accuracy": cpx_before_acc,
            "after_accuracy": cpx_after_acc,
        },
        "threshold": threshold,
        "num_samples": len(bug_labels),
    }
    summary_path = output_dir / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--backbone", default="microsoft/graphcodebert-base")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", default="evaluation/")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        test_data_path=args.test_data,
        backbone_name=args.backbone,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
