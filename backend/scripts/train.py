"""
Training script for CodeLens (Improved)

Features:
  - Multi-task training
  - Mixed precision
  - Gradient accumulation
  - W&B logging
  - Cosine LR schedule
  - Best model checkpointing
  - ✅ Early stopping on validation F1
"""

import argparse
import torch
import wandb
import numpy as np
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score, roc_auc_score
from src.model.model import CodeLensModel, BUG_TYPES
from src.data.dataset import build_dataloaders


# =====================================================
# METRICS
# =====================================================

def compute_metrics(all_bug_preds, all_bug_labels, all_severity_preds, all_severity_labels):
    bug_preds = np.array(all_bug_preds)
    bug_labels = np.array(all_bug_labels)
    bug_binary = (bug_preds >= 0.5).astype(int)

    f1_per_class = []
    auc_per_class = []

    for i in range(bug_labels.shape[1]):
        if bug_labels[:, i].sum() == 0:
            continue
        f1_per_class.append(
            f1_score(bug_labels[:, i], bug_binary[:, i], zero_division=0)
        )
        try:
            auc_per_class.append(
                roc_auc_score(bug_labels[:, i], bug_preds[:, i])
            )
        except:
            pass

    f1_macro = np.mean(f1_per_class) if f1_per_class else 0.0
    auc_mean = np.mean(auc_per_class) if auc_per_class else 0.0

    sev_preds = np.array(all_severity_preds)
    sev_labels = np.array(all_severity_labels)
    severity_mae = float(np.mean(np.abs(sev_preds - sev_labels)))

    return {
        "f1_macro": f1_macro,
        "auc_mean": auc_mean,
        "severity_mae": severity_mae,
    }


# =====================================================
# TRAIN
# =====================================================

def train_epoch(model, loader, optimizer, scheduler, scaler, device, grad_accum_steps, epoch):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]")

    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        cfg_features = batch["cfg_features"].to(device)
        labels_bug = batch["labels_bug"].to(device)
        labels_severity = batch["labels_severity"].to(device)
        labels_cpx_before = batch["labels_complexity_before"].to(device)
        labels_cpx_after = batch["labels_complexity_after"].to(device)

        with autocast():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                cfg_features=cfg_features,
                labels_bug=labels_bug,
                labels_severity=labels_severity,
                labels_complexity_before=labels_cpx_before,
                labels_complexity_after=labels_cpx_after,
            )
            loss = output.loss / grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps
        pbar.set_postfix({
            "loss": f"{loss.item() * grad_accum_steps:.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}"
        })

    return total_loss / len(loader)


# =====================================================
# EVAL
# =====================================================

@torch.no_grad()
def eval_epoch(model, loader, device, epoch):
    model.eval()
    total_loss = 0.0

    all_bug_preds, all_bug_labels = [], []
    all_sev_preds, all_sev_labels = [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [eval]")

    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        cfg_features = batch["cfg_features"].to(device)
        labels_bug = batch["labels_bug"].to(device)
        labels_severity = batch["labels_severity"].to(device)
        labels_cpx_before = batch["labels_complexity_before"].to(device)
        labels_cpx_after = batch["labels_complexity_after"].to(device)

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cfg_features=cfg_features,
            labels_bug=labels_bug,
            labels_severity=labels_severity,
            labels_complexity_before=labels_cpx_before,
            labels_complexity_after=labels_cpx_after,
        )

        total_loss += output.loss.item()

        all_bug_preds.extend(output.bug_probs.cpu().numpy())
        all_bug_labels.extend(labels_bug.cpu().numpy())
        all_sev_preds.extend(output.severity_score.cpu().numpy())
        all_sev_labels.extend(labels_severity.cpu().numpy())

    metrics = compute_metrics(all_bug_preds, all_bug_labels, all_sev_preds, all_sev_labels)
    metrics["loss"] = total_loss / len(loader)

    return metrics


# =====================================================
# MAIN
# =====================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    wandb.init(
        project=cfg.logging.project,
        entity=cfg.logging.entity,
        tags=cfg.logging.tags,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    train_loader, val_loader, _ = build_dataloaders(
        train_path=cfg.data.train_file,
        val_path=cfg.data.val_file,
        test_path=cfg.data.test_file,
        tokenizer_name=cfg.model.backbone,
        batch_size=cfg.training.batch_size,
        max_length=cfg.model.max_seq_length,
    )

    model = CodeLensModel(
        backbone_name=cfg.model.backbone,
        dropout=cfg.model.dropout,
        loss_weights=dict(cfg.training.loss_weights),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    total_steps = len(train_loader) * cfg.training.num_epochs // cfg.training.grad_accumulation_steps
    warmup_steps = int(total_steps * cfg.training.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = GradScaler(enabled=cfg.training.fp16)

    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================
    # EARLY STOPPING
    # ============================

    best_f1 = 0.0
    patience = 3
    patience_counter = 0

    for epoch in range(1, cfg.training.num_epochs + 1):

        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            scaler, device, cfg.training.grad_accumulation_steps, epoch
        )

        val_metrics = eval_epoch(model, val_loader, device, epoch)
        val_f1 = val_metrics["f1_macro"]

        print(f"\nEpoch {epoch}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss  : {val_metrics['loss']:.4f}")
        print(f"F1 Macro  : {val_f1:.4f}")
        print(f"AUC       : {val_metrics['auc_mean']:.4f}")

        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_metrics["loss"],
            "val/f1_macro": val_f1,
            "val/auc_mean": val_metrics["auc_mean"],
        })

        # Save checkpoint
        torch.save(model.state_dict(), output_dir / f"checkpoint_epoch_{epoch}.pt")

        # Early stopping logic
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"✓ New best model saved (F1={best_f1:.4f})")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("\n🚀 Early stopping triggered.")
            break

    print(f"\nTraining complete. Best F1: {best_f1:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()
