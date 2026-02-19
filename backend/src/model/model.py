"""
CodeLens Multi-Task Model.

Architecture:
  - GraphCodeBERT backbone (or CodeBERT) for token + AST embeddings
  - CFG feature fusion via a projection layer
  - Three task heads:
      1. Bug Classification  → multi-label (8 bug types)
      2. Severity Regression → scalar 0–1
      3. Complexity Scoring  → before/after Big-O class prediction

The model outputs logits for all heads simultaneously and is trained
with a weighted multi-task loss.
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from dataclasses import dataclass
from typing import Optional


BUG_TYPES = [
    "security",       # injection, auth bypass, insecure deserialization
    "performance",    # O(n²), inefficient data structures
    "memory",         # leaks, unbounded growth
    "logic",          # wrong conditions, off-by-one
    "type_error",     # wrong type usage, missing casts
    "null_deref",     # unhandled None/null
    "concurrency",    # race conditions, deadlocks
    "style",          # DRY violations, dead code
]

COMPLEXITY_CLASSES = [
    "O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n²)", "O(n³)", "O(2^n)"
]

CFG_FEATURE_DIM = 7     # matches CFGBuilder.features_to_vector output


@dataclass
class CodeLensOutput:
    bug_logits: torch.Tensor            # (batch, num_bug_types) — raw logits
    bug_probs: torch.Tensor             # (batch, num_bug_types) — after sigmoid
    severity_score: torch.Tensor        # (batch,) — 0 to 1
    complexity_before_logits: torch.Tensor  # (batch, num_complexity_classes)
    complexity_after_logits: torch.Tensor   # (batch, num_complexity_classes)
    hidden_states: torch.Tensor         # (batch, seq_len, hidden) — for explainability
    attention_weights: torch.Tensor     # (batch, heads, seq_len, seq_len) — for viz
    loss: Optional[torch.Tensor] = None


class CFGFusionLayer(nn.Module):
    """
    Projects CFG feature vector into transformer hidden space
    and fuses it with the [CLS] token embedding via cross-attention.
    """
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

    def forward(self, cls_embedding: torch.Tensor, cfg_features: torch.Tensor) -> torch.Tensor:
        """
        cls_embedding: (batch, hidden_size)
        cfg_features:  (batch, cfg_dim)
        Returns: fused (batch, hidden_size)
        """
        cfg_proj = self.proj(cfg_features)                   
        gate_input = torch.cat([cls_embedding, cfg_proj], dim=-1)
        gate = torch.sigmoid(self.gate(gate_input))           
        fused = gate * cfg_proj + (1 - gate) * cls_embedding  
        return self.layer_norm(fused)


class BugClassificationHead(nn.Module):
    """Multi-label classifier — one sigmoid per bug type."""
    def __init__(self, hidden_size: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)  


class SeverityHead(nn.Module):
    """Regression head → single scalar 0–1 (0=clean, 1=critical)."""
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(x).squeeze(-1)


class ComplexityHead(nn.Module):
    """
    Predicts Big-O class before and after optimization.
    Two separate classification heads sharing a feature extractor.
    """
    def __init__(self, hidden_size: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
        )
        self.before_head = nn.Linear(256, num_classes)
        self.after_head = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(x)
        return self.before_head(features), self.after_head(features)


class CodeLensModel(nn.Module):
    """
    Full CodeLens multi-task model.

    Inputs:
        input_ids       : (batch, seq_len)  — tokenized code
        attention_mask  : (batch, seq_len)
        token_type_ids  : (batch, seq_len)  — optional
        cfg_features    : (batch, 7)        — CFG numeric features
        ast_input_ids   : (batch, ast_len)  — optional AST token sequence
        labels_bug      : (batch, 8)        — multi-hot bug labels
        labels_severity : (batch,)          — severity float 0–1
        labels_complexity_before : (batch,) — complexity class index
        labels_complexity_after  : (batch,) — complexity class index
    """

    def __init__(
        self,
        backbone_name: str = "microsoft/graphcodebert-base",
        num_bug_classes: int = len(BUG_TYPES),
        num_complexity_classes: int = len(COMPLEXITY_CLASSES),
        dropout: float = 0.1,
        loss_weights: Optional[dict] = None,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.config = AutoConfig.from_pretrained(backbone_name, output_attentions=True, output_hidden_states=True)
        self.backbone = AutoModel.from_pretrained(backbone_name, config=self.config)
        hidden_size = self.config.hidden_size

        self.cfg_fusion = CFGFusionLayer(CFG_FEATURE_DIM, hidden_size)
        self.bug_head = BugClassificationHead(hidden_size, num_bug_classes, dropout)
        self.severity_head = SeverityHead(hidden_size, dropout)
        self.complexity_head = ComplexityHead(hidden_size, num_complexity_classes, dropout)

        self.loss_weights = loss_weights or {
            "bug_classification": 1.0,
            "severity_regression": 0.5,
            "complexity_regression": 0.5,
        }

        self.bug_loss_fn = nn.BCEWithLogitsLoss()
        self.severity_loss_fn = nn.MSELoss()
        self.complexity_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cfg_features: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels_bug: Optional[torch.Tensor] = None,
        labels_severity: Optional[torch.Tensor] = None,
        labels_complexity_before: Optional[torch.Tensor] = None,
        labels_complexity_after: Optional[torch.Tensor] = None,
    ) -> CodeLensOutput:

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # (batch, seq_len, hidden)
        hidden_states = outputs.last_hidden_state
        # (batch, hidden) — [CLS] token
        cls_embedding = hidden_states[:, 0, :]
        # (batch, num_heads, seq_len, seq_len)
        attention_weights = outputs.attentions[-1] if outputs.attentions else None

        # Fuse CFG features
        fused_embedding = self.cfg_fusion(cls_embedding, cfg_features)

        # Task heads
        bug_logits = self.bug_head(fused_embedding)
        bug_probs = torch.sigmoid(bug_logits)
        severity_score = self.severity_head(fused_embedding)
        complexity_before_logits, complexity_after_logits = self.complexity_head(fused_embedding)

        # Loss computation (only when labels provided — training mode)
        loss = None
        if labels_bug is not None:
            loss_bug = self.bug_loss_fn(bug_logits, labels_bug.float())
            loss = self.loss_weights["bug_classification"] * loss_bug

        if labels_severity is not None:
            loss_sev = self.severity_loss_fn(severity_score, labels_severity.float())
            loss = (loss or 0) + self.loss_weights["severity_regression"] * loss_sev

        if labels_complexity_before is not None and labels_complexity_after is not None:
            loss_cpx = (
                self.complexity_loss_fn(complexity_before_logits, labels_complexity_before) +
                self.complexity_loss_fn(complexity_after_logits, labels_complexity_after)
            ) / 2
            loss = (loss or 0) + self.loss_weights["complexity_regression"] * loss_cpx

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

    def get_bug_predictions(self, bug_probs: torch.Tensor, threshold: float = 0.5) -> list[list[str]]:
        """Convert sigmoid probabilities to bug type labels."""
        predictions = []
        for probs in bug_probs:
            detected = [BUG_TYPES[i] for i, p in enumerate(probs) if p >= threshold]
            predictions.append(detected)
        return predictions

    def get_complexity_prediction(self, logits: torch.Tensor) -> list[str]:
        """Convert complexity logits to Big-O string labels."""
        indices = torch.argmax(logits, dim=-1)
        return [COMPLEXITY_CLASSES[i] for i in indices.tolist()]
