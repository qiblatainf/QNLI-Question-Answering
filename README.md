# QNLI Question Answering with Transformers

This project provides a complete pipeline for training, evaluating, and running inference on the GLUE QNLI (Question Natural Language Inference) task using HuggingFace Transformers. It includes scripts for model training, inference, and generating predictions for GLUE submission.

## Overview

- **Task:** Question-answering NLI (QNLI) from the GLUE benchmark
- **Model:** DistilBERT fine-tuned for sequence classification
- **Frameworks:** PyTorch, HuggingFace Transformers, Datasets
- **Data:** GLUE QNLI and SQuAD-format datasets

## Directory Structure

```
qnli-question-answering/
├── src/
│   └── qnli_question_answering/
│       ├── main.py           # Training and evaluation script
│       ├── inference.py      # Inference and prediction script
│       ├── predictions.py    # SQuAD-format prediction script
│       └── results/          # Stores results and logs
│          ├── qnli_test_predictions.tsv # Final predictions for GLUE submission
├── dataset/                  # Contains train-v2.0.json, dev-v2.0.json (downloadable from SQuAD website)
├── qnli_entailment_model/    # Saved fine-tuned model and tokenizer
├── results-qnli-entailment/  # Training outputs and checkpoints
├── .gitignore
└── README.md
```

## Dependencies

- `datasets`
- `torch`
- `transformers`
- `scipy`
- `scikit-learn`
- `matplotlib`

Install all dependencies with:
```bash
pip install datasets torch transformers scipy scikit-learn matplotlib
```

## Usage

### 1. Training & Evaluation

Run the main training script:
```bash
python src/qnli_question_answering/main.py
```
- Downloads and tokenizes the GLUE QNLI dataset
- Fine-tunes DistilBERT
- Evaluates on the validation set
- Saves the model and tokenizer to `qnli_entailment_model/`
- Outputs results to `results-qnli-entailment/` and `src/qnli_question_answering/results/`

### 2. Inference (GLUE Submission Format)

Run inference on the test set and save predictions:
```bash
python src/qnli_question_answering/inference.py \
  --model_dir qnli_entailment_model \
  --output qnli_test_predictions.tsv
```
- Supports additional arguments: `--task`, `--split`, `--max_length`, `--batch_size`

### 3. SQuAD-format Predictions

Generate predictions for SQuAD-format data:
```bash
python src/qnli_question_answering/predictions.py
```
- Reads `dataset/dev-v2.0.json`
- Outputs predictions to `qnli_test_predictions.tsv`

## Data

- Place SQuAD-format files (`train-v2.0.json`, `dev-v2.0.json`) in the `dataset/` directory.
- The GLUE QNLI dataset is automatically downloaded by the scripts.

## Results & Model

- Fine-tuned models are saved in `qnli_entailment_model/`
- Training logs and checkpoints in `results-qnli-entailment/`
- Validation results in `src/qnli_question_answering/results/`
- Final predictions in `qnli_test_predictions.tsv`

## Notes

- The `.gitignore` excludes large data and cache directories.
- For GPU acceleration, ensure PyTorch is installed with CUDA support.
- For custom datasets or tasks, modify the scripts as needed.

*Created with HuggingFace Transformers and the GLUE benchmark.*