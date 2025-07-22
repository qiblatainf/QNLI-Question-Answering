#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from torch.utils.data import Dataset

class InferenceDataset(Dataset):
    """wrapping tokenized encodings for inference."""
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return self.encodings['input_ids'].size(0)

    def __getitem__(self, idx):
        return {key: tensor[idx] for key, tensor in self.encodings.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on a fine-tuned model and generate a GLUE-formatted TSV submission"
    )
    parser.add_argument(
        "--model_dir", type=str, required=True,
        help="Path to your exported model directory"
    )
    parser.add_argument(
        "--task", type=str, default="qnli",
        help="GLUE task name (e.g., qnli, cola, mrpc, etc.)"
    )
    parser.add_argument(
        "--split", type=str, default="test",
        help="Dataset split to run on (default: test)"
    )
    parser.add_argument(
        "--max_length", type=int, default=64,
        help="Max sequence length for tokenization"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to output TSV file (one label per line, no header)"
    )
    args = parser.parse_args()

    # Load the GLUE dataset
    raw_dataset = load_dataset("glue", args.task)
    df = raw_dataset[args.split].to_pandas()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

    # Tokenize examples
    questions = df.get("question", df.get("sentence1")).astype(str).tolist()
    sentences = df.get("sentence", df.get("sentence2")).astype(str).tolist()
    encodings = tokenizer(
        questions,
        sentences,
        truncation=True,
        padding="max_length",
        max_length=args.max_length,
        return_tensors="pt"
    )

    # Prepare inference dataset
    test_dataset = InferenceDataset(encodings)

    # Run prediction
    trainer = Trainer(model=model)
    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions
    preds = np.argmax(logits, axis=-1)

    # Save predictions to TSV
    pd.DataFrame(preds).to_csv(args.output, sep="\t", index=False, header=False)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()


