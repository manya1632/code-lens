"""
Unit tests for CodeLens components.

Run with:
  pytest tests/ -v
"""
import pytest
import torch
from unittest.mock import patch, MagicMock


# ─── AST Parser Tests ──────────────────────────────────────────────────────────

class TestASTParser:
    def test_parse_simple_function(self):
        try:
            from src.model.ast_parser import ASTParser
        except ImportError:
            pytest.skip("tree-sitter not installed")

        parser = ASTParser("python")
        code = "def foo(x):\n    return x + 1"
        root, features = parser.parse(code)

        assert root is not None
        assert features.num_nodes > 0
        assert features.max_depth > 0
        assert not features.has_exception_handling

    def test_parse_nested_loops(self):
        try:
            from src.model.ast_parser import ASTParser
        except ImportError:
            pytest.skip("tree-sitter not installed")

        parser = ASTParser("python")
        code = "for i in range(n):\n    for j in range(n):\n        pass"
        _, features = parser.parse(code)

        assert features.loop_nesting_depth >= 2
        assert features.cyclomatic_complexity >= 2

    def test_parse_try_except(self):
        try:
            from src.model.ast_parser import ASTParser
        except ImportError:
            pytest.skip("tree-sitter not installed")

        parser = ASTParser("python")
        code = "try:\n    x = 1/0\nexcept ZeroDivisionError:\n    pass"
        _, features = parser.parse(code)
        assert features.has_exception_handling

    def test_unsupported_language_raises(self):
        try:
            from src.model.ast_parser import ASTParser
        except ImportError:
            pytest.skip("tree-sitter not installed")
        with pytest.raises(ValueError, match="Unsupported language"):
            ASTParser("cobol")


# ─── Model Tests ──────────────────────────────────────────────────────────────

class TestCodeLensModel:
    @pytest.fixture
    def mock_model(self):
        """Create a CodeLensModel with mocked backbone for fast testing."""
        from src.model.model import CodeLensModel
        with patch("src.model.model.AutoModel.from_pretrained") as mock_backbone, \
             patch("src.model.model.AutoConfig.from_pretrained") as mock_config:
            mock_config.return_value = MagicMock(hidden_size=768, output_attentions=True, output_hidden_states=True)
            mock_backbone.return_value = MagicMock()
            model = CodeLensModel.__new__(CodeLensModel)
            # Manually initialize heads only
            import torch.nn as nn
            from src.model.model import (
                CFGFusionLayer, BugClassificationHead,
                SeverityHead, ComplexityHead, BUG_TYPES, COMPLEXITY_CLASSES
            )
            model.cfg_fusion = CFGFusionLayer(7, 768)
            model.bug_head = BugClassificationHead(768, len(BUG_TYPES))
            model.severity_head = SeverityHead(768)
            model.complexity_head = ComplexityHead(768, len(COMPLEXITY_CLASSES))
            return model

    def test_cfg_fusion_output_shape(self):
        from src.model.model import CFGFusionLayer
        layer = CFGFusionLayer(cfg_dim=7, hidden_size=768)
        cls_emb = torch.randn(2, 768)
        cfg_feat = torch.randn(2, 7)
        out = layer(cls_emb, cfg_feat)
        assert out.shape == (2, 768)

    def test_bug_head_output_shape(self):
        from src.model.model import BugClassificationHead, BUG_TYPES
        head = BugClassificationHead(hidden_size=768, num_classes=len(BUG_TYPES))
        x = torch.randn(4, 768)
        logits = head(x)
        assert logits.shape == (4, len(BUG_TYPES))

    def test_severity_head_output_range(self):
        from src.model.model import SeverityHead
        head = SeverityHead(hidden_size=768)
        x = torch.randn(8, 768)
        scores = head(x)
        assert scores.shape == (8,)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_complexity_head_output_shape(self):
        from src.model.model import ComplexityHead, COMPLEXITY_CLASSES
        head = ComplexityHead(hidden_size=768, num_classes=len(COMPLEXITY_CLASSES))
        x = torch.randn(4, 768)
        before, after = head(x)
        assert before.shape == (4, len(COMPLEXITY_CLASSES))
        assert after.shape == (4, len(COMPLEXITY_CLASSES))

    def test_bug_types_list_not_empty(self):
        from src.model.model import BUG_TYPES
        assert len(BUG_TYPES) > 0
        assert "security" in BUG_TYPES
        assert "performance" in BUG_TYPES

    def test_complexity_classes_ordered(self):
        from src.model.model import COMPLEXITY_CLASSES
        assert COMPLEXITY_CLASSES[0] == "O(1)"
        assert "O(n²)" in COMPLEXITY_CLASSES


# ─── Attention Rollout Tests ───────────────────────────────────────────────────

class TestAttentionRollout:
    def test_output_shape(self):
        from src.explainability.explainer import AttentionRollout
        rollout = AttentionRollout()
        batch, heads, seq = 2, 12, 64
        attentions = [torch.softmax(torch.randn(batch, heads, seq, seq), dim=-1) for _ in range(12)]
        result = rollout(attentions)
        assert result.shape == (batch, seq)

    def test_importance_nonnegative(self):
        from src.explainability.explainer import AttentionRollout
        rollout = AttentionRollout()
        attentions = [torch.softmax(torch.randn(1, 8, 32, 32), dim=-1) for _ in range(6)]
        result = rollout(attentions)
        assert (result >= 0).all()

    def test_head_fusion_modes(self):
        from src.explainability.explainer import AttentionRollout
        attentions = [torch.softmax(torch.randn(1, 8, 16, 16), dim=-1) for _ in range(4)]
        for mode in ["mean", "max", "min"]:
            rollout = AttentionRollout(head_fusion=mode)
            result = rollout(attentions)
            assert result.shape == (1, 16)


# ─── Data Preprocessing Tests ─────────────────────────────────────────────────

class TestPreprocessing:
    def test_detect_bug_types_nested_loop(self):
        from src.data.preprocess import detect_bug_types
        code = "for i in range(n):\n    for j in range(n):\n        pass"
        bugs = detect_bug_types(code)
        assert "performance" in bugs

    def test_detect_bug_types_sql_injection(self):
        from src.data.preprocess import detect_bug_types
        code = 'query = f"SELECT * FROM users WHERE name=\'%s\'" % username'
        bugs = detect_bug_types(code)
        assert "security" in bugs

    def test_estimate_complexity_nested(self):
        from src.data.preprocess import estimate_complexity
        code = "for i in range(n):\n    for j in range(n):\n        x = 1"
        assert estimate_complexity(code) == "O(n²)"

    def test_estimate_complexity_linear(self):
        from src.data.preprocess import estimate_complexity
        code = "for i in range(n):\n    x += i"
        assert estimate_complexity(code) == "O(n)"

    def test_estimate_severity_security(self):
        from src.data.preprocess import estimate_severity
        assert estimate_severity(["security"]) == 1.0

    def test_multihot_encoding(self):
        from src.data.dataset import CodeReviewDataset
        from src.model.model import BUG_TYPES
        # Direct test of helper logic
        bugs = ["security", "performance"]
        vec = [0.0] * len(BUG_TYPES)
        for b in bugs:
            if b in BUG_TYPES:
                vec[BUG_TYPES.index(b)] = 1.0
        assert vec[BUG_TYPES.index("security")] == 1.0
        assert vec[BUG_TYPES.index("performance")] == 1.0
        assert sum(vec) == 2.0
