"""
CodeLens Multi-Task Model (Improved Version)

Enhancements:
- Proper class-balanced BCE loss
- Device-safe pos_weight handling
- Stable multi-task loss accumulation
- Cleaner loss logic
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from dataclasses import dataclass
from typing import Optional

# -----------------------------
# Bug & Complexity Definitions
# -----------------------------

BUG_TYPES = [
    "security",
    "performance",
    "memory",
    "logic",
    "type_error",
    "null_deref",
    "concurrency",
    "style",
]

COMPLEXITY_CLASSES = [
    "O(1)", "O(log n)", "O(n)", "O(n log n)",
    "O(n²)", "O(n³)", "O(2^n)"
]

CFG_FEATURE_DIM = 7


# -----------------------------
# Output Container
# -----------------------------

@dataclass
class CodeLensOutput:
    bug_logits: torch.Tensor
    bug_probs: torch.Tensor
    severity_score: torch.Tensor
    complexity_before_logits: torch.Tensor
    complexity_after_logits: torch.Tensor
    hidden_states: torch.Tensor
    attention_weights: Optional[torch.Tensor]
    loss: Optional[torch.Tensor] = None


# -----------------------------
# CFG Fusion
# -----------------------------

class CFGFusionLayer(nn.Module):
    def __init__(self, cfg_dim: int, hidden_size: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(cfg_dim, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        self.gate = nn.Linear(hidden_size * 2, 1)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, cls_embedding, cfg_features):
        cfg_proj = self.proj(cfg_features)
        gate_input = torch.cat([cls_embedding, cfg_proj], dim=-1)
        gate = torch.sigmoid(self.gate(gate_input))
        fused = gate * cfg_proj + (1 - gate) * cls_embedding
        return self.layer_norm(fused)


# -----------------------------
# Task Heads
# -----------------------------

class BugClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_classes, dropout=0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


class SeverityHead(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.regressor(x).squeeze(-1)


class ComplexityHead(nn.Module):
    def __init__(self, hidden_size, num_classes, dropout=0.1):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
        )
        self.before_head = nn.Linear(256, num_classes)
        self.after_head = nn.Linear(256, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.before_head(features), self.after_head(features)


# -----------------------------
# Main Model
# -----------------------------

class CodeLensModel(nn.Module):

    def __init__(
        self,
        backbone_name="microsoft/graphcodebert-base",
        num_bug_classes=len(BUG_TYPES),
        num_complexity_classes=len(COMPLEXITY_CLASSES),
        dropout=0.1,
        loss_weights=None,
    ):
        super().__init__()

        self.config = AutoConfig.from_pretrained(
            backbone_name,
            output_attentions=True,
            output_hidden_states=True
        )
        self.backbone = AutoModel.from_pretrained(backbone_name, config=self.config)
        hidden_size = self.config.hidden_size

        # Heads
        self.cfg_fusion = CFGFusionLayer(CFG_FEATURE_DIM, hidden_size)
        self.bug_head = BugClassificationHead(hidden_size, num_bug_classes, dropout)
        self.severity_head = SeverityHead(hidden_size, dropout)
        self.complexity_head = ComplexityHead(hidden_size, num_complexity_classes, dropout)

        self.loss_weights = loss_weights or {
            "bug_classification": 1.0,
            "severity_regression": 0.5,
            "complexity_regression": 0.5,
        }

        # -----------------------------
        # Class-Balanced BCE Loss
        # -----------------------------

        # Example distribution (adjust if needed)
        class_counts = torch.tensor([160, 140, 110, 90, 60, 40, 80, 120], dtype=torch.float)
        total = class_counts.sum()
        pos_weights = total / class_counts

        # REGISTER AS BUFFER → auto moves with model.to(device)
        self.register_buffer("pos_weights", pos_weights)

        self.bug_loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weights)
        self.severity_loss_fn = nn.MSELoss()
        self.complexity_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    # -----------------------------
    # Forward
    # -----------------------------

    def forward(
        self,
        input_ids,
        attention_mask,
        cfg_features,
        token_type_ids=None,
        labels_bug=None,
        labels_severity=None,
        labels_complexity_before=None,
        labels_complexity_after=None,
    ):

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        hidden_states = outputs.last_hidden_state
        cls_embedding = hidden_states[:, 0, :]
        attention_weights = outputs.attentions[-1] if outputs.attentions else None

        fused_embedding = self.cfg_fusion(cls_embedding, cfg_features)

        bug_logits = self.bug_head(fused_embedding)
        bug_probs = torch.sigmoid(bug_logits)
        severity_score = self.severity_head(fused_embedding)
        complexity_before_logits, complexity_after_logits = self.complexity_head(fused_embedding)

        # -----------------------------
        # Multi-task Loss
        # -----------------------------

        loss = None

        if labels_bug is not None:
            loss_bug = self.bug_loss_fn(bug_logits, labels_bug.float())
            loss = self.loss_weights["bug_classification"] * loss_bug

        if labels_severity is not None:
            loss_sev = self.severity_loss_fn(severity_score, labels_severity.float())
            loss = (loss if loss is not None else 0) + \
                   self.loss_weights["severity_regression"] * loss_sev

        if labels_complexity_before is not None and labels_complexity_after is not None:
            loss_cpx = (
                self.complexity_loss_fn(complexity_before_logits, labels_complexity_before) +
                self.complexity_loss_fn(complexity_after_logits, labels_complexity_after)
            ) / 2

            loss = (loss if loss is not None else 0) + \
                   self.loss_weights["complexity_regression"] * loss_cpx

        return CodeLensOutput(
            bug_logits=bug_logits,
            bug_probs=bug_probs,
            severity_score=severity_score,
            complexity_before_logits=complexity_before_logits,
            complexity_after_logits=complexity_after_logits,
            hidden_states=hidden_states,
            attention_weights=attention_weights,
            loss=loss,
        )
    