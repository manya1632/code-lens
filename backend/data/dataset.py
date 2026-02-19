"""
Dataset classes for CodeLens training.

Supports two dataset formats:
  1. Bugs2Fix format  — (buggy_code, fixed_code, bug_type)
  2. CodeSearchNet    — (code, docstring, language) for pre-training
  3. Custom JSONL     — unified format for fine-tuning

JSONL format expected:
{
  "code": "def foo(x):\n    ...",
  "language": "python",
  "bugs": ["performance", "logic"],          // multi-label
  "severity": 0.8,                           // float 0–1
  "complexity_before": "O(n²)",
  "complexity_after": "O(n)",
  "fixed_code": "def foo(x):\n    ..."       // optional, for future seq2seq
}
"""
import json
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Optional
from src.model.ast_parser import ASTParser
from src.model.cfg_builder import CFGBuilder
from src.model.model import BUG_TYPES, COMPLEXITY_CLASSES


COMPLEXITY_TO_IDX = {c: i for i, c in enumerate(COMPLEXITY_CLASSES)}


class CodeReviewDataset(Dataset):

    def __init__(
        self,
        data_path: str,
        tokenizer_name: str = "microsoft/graphcodebert-base",
        max_length: int = 512,
        use_ast: bool = True,
        use_cfg: bool = True,
        cache_features: bool = True,
    ):
        self.data_path = Path(data_path)
        self.max_length = max_length
        self.use_ast = use_ast
        self.use_cfg = use_cfg

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.samples = self._load_samples()

        # Pre-built parsers per language
        self._ast_parsers: dict[str, ASTParser] = {}
        self._cfg_builders: dict[str, CFGBuilder] = {}

        # Optional feature cache to avoid recomputing AST/CFG
        self._feature_cache: dict = {} if cache_features else None

    def _load_samples(self) -> list[dict]:
        samples = []
        with open(self.data_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    def _get_ast_parser(self, language: str) -> Optional[ASTParser]:
        if language not in self._ast_parsers:
            try:
                self._ast_parsers[language] = ASTParser(language)
            except ValueError:
                return None
        return self._ast_parsers[language]

    def _get_cfg_builder(self, language: str) -> Optional[CFGBuilder]:
        if language not in self._cfg_builders:
            try:
                self._cfg_builders[language] = CFGBuilder(language)
            except Exception:
                return None
        return self._cfg_builders[language]

    def _extract_cfg_features(self, code: str, language: str) -> list[float]:
        builder = self._get_cfg_builder(language)
        if builder is None:
            return [0.0] * 7
        try:
            _, features = builder.build(code)
            return builder.features_to_vector(features)
        except Exception:
            return [0.0] * 7

    def _bugs_to_multihot(self, bugs: list[str]) -> list[float]:
        vec = [0.0] * len(BUG_TYPES)
        for bug in bugs:
            if bug in BUG_TYPES:
                vec[BUG_TYPES.index(bug)] = 1.0
        return vec

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        if self._feature_cache is not None and idx in self._feature_cache:
            return self._feature_cache[idx]

        sample = self.samples[idx]
        code = sample["code"]
        language = sample.get("language", "python")

        # Tokenize code
        encoding = self.tokenizer(
            code,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # CFG features
        cfg_features = self._extract_cfg_features(code, language) if self.use_cfg else [0.0] * 7

        # Labels
        bugs = sample.get("bugs", [])
        bug_labels = self._bugs_to_multihot(bugs)

        severity = float(sample.get("severity", 0.0))

        complexity_before_str = sample.get("complexity_before", "O(n)")
        complexity_after_str = sample.get("complexity_after", "O(n)")
        complexity_before = COMPLEXITY_TO_IDX.get(complexity_before_str, 2)
        complexity_after = COMPLEXITY_TO_IDX.get(complexity_after_str, 2)

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "cfg_features": torch.tensor(cfg_features, dtype=torch.float),
            "labels_bug": torch.tensor(bug_labels, dtype=torch.float),
            "labels_severity": torch.tensor(severity, dtype=torch.float),
            "labels_complexity_before": torch.tensor(complexity_before, dtype=torch.long),
            "labels_complexity_after": torch.tensor(complexity_after, dtype=torch.long),
            # Metadata for inference
            "code": code,
            "language": language,
            "fixed_code": sample.get("fixed_code", ""),
        }

        if self._feature_cache is not None:
            self._feature_cache[idx] = item

        return item


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate — handles string fields separately from tensors."""
    tensor_keys = ["input_ids", "attention_mask", "cfg_features",
                   "labels_bug", "labels_severity",
                   "labels_complexity_before", "labels_complexity_after"]
    str_keys = ["code", "language", "fixed_code"]

    collated = {}
    for key in tensor_keys:
        collated[key] = torch.stack([item[key] for item in batch])
    for key in str_keys:
        collated[key] = [item[key] for item in batch]
    return collated


def build_dataloaders(
    train_path: str,
    val_path: str,
    test_path: str,
    tokenizer_name: str,
    batch_size: int = 16,
    max_length: int = 512,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    kwargs = dict(tokenizer_name=tokenizer_name, max_length=max_length)
    train_ds = CodeReviewDataset(train_path, **kwargs)
    val_ds = CodeReviewDataset(val_path, **kwargs)
    test_ds = CodeReviewDataset(test_path, **kwargs)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def stratified_split_jsonl(input_path, train_path, val_path, test_path):
    """
    Performs stratified split on bug labels.
    """

    samples = []
    with open(input_path, "r") as f:
        for line in f:
            samples.append(json.loads(line))

    # Convert multi-label to single stratification label
    # (use first bug type for stratification)
    stratify_labels = [
        sample["bugs"][0] if sample["bugs"] else "none"
        for sample in samples
    ]

    train_val, test = train_test_split(
        samples,
        test_size=0.1,
        stratify=stratify_labels,
        random_state=42
    )

    stratify_train = [
        sample["bugs"][0] if sample["bugs"] else "none"
        for sample in train_val
    ]

    train, val = train_test_split(
        train_val,
        test_size=0.1,
        stratify=stratify_train,
        random_state=42
    )

    def write(path, data):
        with open(path, "w") as f:
            for sample in data:
                f.write(json.dumps(sample) + "\n")

    write(train_path, train)
    write(val_path, val)
    write(test_path, test)

    print("Stratified split completed.")
