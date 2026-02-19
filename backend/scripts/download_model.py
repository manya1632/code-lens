"""
Downloads GraphCodeBERT backbone and tokenizer from HuggingFace Hub.
Run this before training.

Usage:
  python scripts/download_model.py
  python scripts/download_model.py --model microsoft/codebert-base
"""
import argparse
from transformers import AutoModel, AutoTokenizer
from pathlib import Path


def download(model_name: str, save_dir: str):
    save_path = Path(save_dir) / model_name.replace("/", "_")
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(str(save_path))

    print(f"Downloading model: {model_name}")
    model = AutoModel.from_pretrained(model_name)
    model.save_pretrained(str(save_path))

    print(f"Saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/graphcodebert-base")
    parser.add_argument("--save_dir", default="./pretrained")
    args = parser.parse_args()
    download(args.model, args.save_dir)


if __name__ == "__main__":
    main()
