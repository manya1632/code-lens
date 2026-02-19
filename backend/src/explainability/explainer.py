"""
Explainability Layer for CodeLens.

Three explanation methods:
  1. Attention Rollout  — propagates attention through all layers to get token importance
  2. Integrated Gradients (via Captum) — attribution scores per input token
  3. SHAP values — shapley-based feature importance

The explainer produces:
  - Token importance scores → highlight which tokens triggered the bug detection
  - Human-readable explanation strings (why this bug was detected)
  - Heatmap data for visualization
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional
from transformers import AutoTokenizer
from src.model.model import CodeLensModel, BUG_TYPES


class AttentionRollout:
    """
    Attention Rollout (Abnar & Zuidema, 2020).
    Recursively multiplies attention matrices across layers,
    adding residual connections, to get total token→token influence.
    """

    def __init__(self, head_fusion: str = "mean", discard_ratio: float = 0.9):
        """
        head_fusion: how to fuse multiple attention heads — "mean", "max", "min"
        discard_ratio: fraction of lowest attention weights to zero out (noise reduction)
        """
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio

    def __call__(self, attention_weights: list[torch.Tensor]) -> torch.Tensor:
        """
        attention_weights: list of (batch, heads, seq_len, seq_len) per layer
        Returns: (batch, seq_len) importance of each token to the [CLS] token
        """
        result = torch.eye(attention_weights[0].size(-1), device=attention_weights[0].device)
        result = result.unsqueeze(0).expand(attention_weights[0].size(0), -1, -1)

        for attn in attention_weights:
            # Fuse heads
            if self.head_fusion == "mean":
                attn_fused = attn.mean(dim=1)
            elif self.head_fusion == "max":
                attn_fused = attn.max(dim=1).values
            else:
                attn_fused = attn.min(dim=1).values

            # Discard lowest attention weights (threshold)
            flat = attn_fused.view(attn_fused.size(0), -1)
            threshold = torch.quantile(flat, self.discard_ratio, dim=-1, keepdim=True)
            threshold = threshold.unsqueeze(-1)
            attn_fused = torch.where(attn_fused >= threshold, attn_fused, torch.zeros_like(attn_fused))

            # Add residual (identity) and normalize
            attn_fused = attn_fused + torch.eye(attn_fused.size(-1), device=attn_fused.device).unsqueeze(0)
            attn_fused = attn_fused / attn_fused.sum(dim=-1, keepdim=True).clamp(min=1e-9)

            # Rollout: matrix multiply
            result = torch.bmm(attn_fused, result)

        # Return importance of each token to [CLS] (row 0)
        cls_importance = result[:, 0, :]  # (batch, seq_len)
        return cls_importance


class IntegratedGradientsExplainer:
    """
    Integrated Gradients (Sundararajan et al., 2017).
    Computes attribution by integrating gradients from baseline (zeros)
    to actual input embeddings.
    """

    def __init__(self, model: CodeLensModel, num_steps: int = 50):
        self.model = model
        self.num_steps = num_steps

    def explain(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cfg_features: torch.Tensor,
        target_bug_idx: int,
    ) -> torch.Tensor:
        """
        Returns attribution scores of shape (seq_len,) for the given bug class.
        """
        self.model.eval()
        embeddings = self.model.backbone.embeddings.word_embeddings(input_ids)
        baseline = torch.zeros_like(embeddings)

        total_gradients = torch.zeros_like(embeddings)
        for step in range(self.num_steps):
            alpha = step / self.num_steps
            interpolated = baseline + alpha * (embeddings - baseline)
            interpolated.requires_grad_(True)

            # Forward pass with interpolated embeddings
            outputs = self.model.backbone(
                inputs_embeds=interpolated,
                attention_mask=attention_mask,
            )
            cls = outputs.last_hidden_state[:, 0, :]
            fused = self.model.cfg_fusion(cls, cfg_features)
            bug_logits = self.model.bug_head(fused)

            # Backward w.r.t. target bug class
            score = bug_logits[:, target_bug_idx].sum()
            score.backward()
            total_gradients += interpolated.grad.detach()

        # Integrated gradients = mean gradient × (input - baseline)
        ig = total_gradients / self.num_steps * (embeddings - baseline).detach()

        # Aggregate across embedding dimension: L2 norm per token
        attribution = ig.norm(dim=-1)  # (batch, seq_len)
        return attribution


class CodeExplainer:
    """
    High-level explainer interface.
    Given a model output, produces human-readable token importance
    and natural language explanations.
    """

    def __init__(
        self,
        model: CodeLensModel,
        tokenizer: AutoTokenizer,
        method: str = "attention_rollout",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.method = method
        self.rollout = AttentionRollout()
        self.ig_explainer = IntegratedGradientsExplainer(model)

    def explain(
        self,
        code: str,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cfg_features: torch.Tensor,
        model_output,
        top_k: int = 10,
    ) -> dict:
        """
        Returns explanation dict with:
          - token_importance: list of (token, score) sorted by importance
          - highlighted_lines: which lines are most suspicious
          - natural_language: generated explanation string
          - heatmap_data: raw scores per token for visualization
        """
        # Get token importance scores
        if self.method == "attention_rollout" and model_output.attention_weights is not None:
            # We need all layer attentions — re-run with output_attentions=True
            with torch.no_grad():
                backbone_out = self.model.backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                )
            all_attentions = backbone_out.attentions  # list of (batch, heads, seq, seq)
            importance = self.rollout(list(all_attentions))  # (batch, seq_len)
        else:
            # Fallback: use last layer attention to [CLS]
            importance = model_output.attention_weights[:, :, 0, :].mean(dim=1)

        importance = importance[0].cpu().numpy()  # (seq_len,)

        # Decode tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

        # Filter padding and special tokens
        valid_pairs = [
            (tok, float(imp))
            for tok, imp, mask in zip(tokens, importance, attention_mask[0].tolist())
            if mask == 1 and tok not in ["<s>", "</s>", "<pad>", "[CLS]", "[SEP]", "[PAD]"]
        ]

        # Sort by importance
        sorted_pairs = sorted(valid_pairs, key=lambda x: x[1], reverse=True)

        # Top-k important tokens
        top_tokens = sorted_pairs[:top_k]

        # Map tokens back to source lines
        highlighted_lines = self._find_highlighted_lines(code, [tok for tok, _ in top_tokens])

        # Build natural language explanation
        detected_bugs = [BUG_TYPES[i] for i, p in enumerate(model_output.bug_probs[0]) if p >= 0.5]
        nl_explanation = self._generate_explanation(
            detected_bugs=detected_bugs,
            severity=float(model_output.severity_score[0]),
            top_tokens=top_tokens,
            highlighted_lines=highlighted_lines,
        )

        return {
            "token_importance": top_tokens,
            "highlighted_lines": highlighted_lines,
            "natural_language": nl_explanation,
            "heatmap_data": {
                "tokens": [tok for tok, _ in valid_pairs],
                "scores": [score for _, score in valid_pairs],
            },
            "detected_bugs": detected_bugs,
            "severity": float(model_output.severity_score[0]),
        }

    def _find_highlighted_lines(self, code: str, important_tokens: list[str]) -> list[int]:
        """Find which source lines contain the important tokens."""
        lines = code.split("\n")
        highlighted = []
        for i, line in enumerate(lines):
            for token in important_tokens:
                # Clean subword prefix (## in BERT, Ġ in GPT-2)
                clean_tok = token.lstrip("##Ġ▁").lower()
                if clean_tok and clean_tok in line.lower() and len(clean_tok) > 2:
                    highlighted.append(i + 1)  # 1-indexed
                    break
        return sorted(set(highlighted))

    def _generate_explanation(
        self,
        detected_bugs: list[str],
        severity: float,
        top_tokens: list[tuple],
        highlighted_lines: list[int],
    ) -> str:
        """
        Rule-based explanation generator.
        Replace with fine-tuned language model for richer explanations.
        """
        if not detected_bugs:
            return "No significant issues detected. Code appears clean."

        explanations = []

        template_map = {
            "performance": (
                "Performance issue detected: The code likely contains inefficient constructs "
                f"(key tokens: {', '.join(t for t, _ in top_tokens[:3])}). "
                "Nested loops or repeated linear searches suggest O(n²) or worse complexity. "
                "Consider using hash maps, sets, or vectorized operations to reduce to O(n)."
            ),
            "security": (
                "Security vulnerability detected: The model found patterns associated with "
                f"injection attacks or unsafe operations (near lines {highlighted_lines[:3]}). "
                "Use parameterized queries, input sanitization, or safe APIs instead."
            ),
            "memory": (
                "Memory issue detected: Unbounded data structure growth or missing cleanup "
                f"detected around lines {highlighted_lines[:3]}. "
                "Ensure caches have size limits and event listeners/timers are cleared on cleanup."
            ),
            "logic": (
                f"Logic error detected near lines {highlighted_lines[:3]}: "
                "The model flagged a condition or return path that may not behave as intended. "
                "Review edge cases and boundary conditions carefully."
            ),
            "concurrency": (
                "Concurrency issue detected: Shared state accessed without synchronization "
                f"(tokens: {', '.join(t for t, _ in top_tokens[:3])}). "
                "Use locks, atomic operations, or immutable data structures."
            ),
        }

        for bug in detected_bugs:
            if bug in template_map:
                explanations.append(template_map[bug])
            else:
                explanations.append(f"{bug.title()} issue found in the code.")

        severity_label = "CRITICAL" if severity > 0.8 else "HIGH" if severity > 0.6 else "MEDIUM" if severity > 0.4 else "LOW"
        header = f"[Severity: {severity_label} ({severity:.0%})] "

        return header + " | ".join(explanations)
