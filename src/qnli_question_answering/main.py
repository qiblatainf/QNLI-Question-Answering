from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
import evaluate #requires ['scipy', 'scikit-learn']

#Load the QNLI dataset
raw_dataset = load_dataset("glue", "qnli")

#Print the keys of the dataset to see available splits
print("Available splits in the dataset:")
# print(raw_dataset.keys())

#Convert the dataset to pandas DataFrames for easier manipulation
train_df = raw_dataset["train"].to_pandas()
val_df   = raw_dataset["validation"].to_pandas()
test_df  = raw_dataset["test"].to_pandas()

# print("Train DataFrame:")
# print(train_df.head())  

# Initialize tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2) #0 or 1 (non-entailment vs. entailment)

#Tokenize the input data
def tokenize_function(examples: pd.DataFrame):
    questions = examples["question"].astype(str).tolist()
    # print("Questions:", questions[:5])  # Print first 5 questions for debugging
    sentences = examples["sentence"].astype(str).tolist()
    # print("Sentences:", examples["sentence"].astype(str).tolist()[:5])  # Print first 5 sentences for debugging
    return tokenizer(questions, sentences, truncation=True, padding="max_length", max_length=64, return_tensors="pt")

train_encodings = tokenize_function(train_df)
# print("Tokenized Train Encodings:")
# print(train_encodings)

val_encodings = tokenize_function(val_df)   
# print("Tokenized Validation Encodings:")
# print(val_encodings)

test_encodings = tokenize_function(test_df)
# print("Tokenized Test Encodings:")
# print(test_encodings)

# train_df = train_df.sample(frac=0.5, random_state=42).reset_index(drop=True)
# print(f"Subsetted Train DF: {len(train_df)} rows (from {len(train_df)} total)")

class QNLIDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: tensor[idx] for key, tensor in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels.iloc[idx])
        return item

train_dataset = QNLIDataset(train_encodings, train_df['label'])
eval_dataset  = QNLIDataset(val_encodings, val_df['label'])


# 6. Load evaluation metric
metric = evaluate.load("glue", "qnli")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)

# 7. Define training arguments
training_args = TrainingArguments(
    output_dir="./results-qnli-entailment",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=50,
)

# 8. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# 9. Train the model
# trainer.train()

checkpoint_path = "./results-qnli-entailment/checkpoint-500"
trainer.train(resume_from_checkpoint=checkpoint_path)

# 10. Evaluate on validation set
eval_results = trainer.evaluate()
print("Validation results:", eval_results)

# 11. Save the fine-tuned model and tokenizer
model.save_pretrained("./qnli_entailment_model")
tokenizer.save_pretrained("./qnli_entailment_model")