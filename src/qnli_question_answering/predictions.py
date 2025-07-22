import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification
import main  # Import the main script to access functions and variables
import pandas as pd

#Load the fine-tuned model
model_path = "./qnli_entailment_model"  #model path
model = AutoModelForSequenceClassification.from_pretrained(model_path)

#Set the model to the correct device (CPU or GPU)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

#Load dev-v2.0.json
dev_filepath = "dataset\dev-v2.0.json"  #replace with the actual path to your file

with open(dev_filepath, 'r', encoding='utf-8') as f:
    squad_data = json.load(f)

#Extract question-context pairs from SQuAD data
test_examples = []
for group in squad_data['data']:
    for passage in group['paragraphs']:
        context = passage['context']
        for qa in passage['qas']:
            question = qa['question']
            test_examples.append({'question': question, 'sentence': context, 'index': qa['id']})

test_df = pd.DataFrame(test_examples)

#Tokenize the test data
test_encodings = main.tokenize_function(test_df)

class CustomDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

test_dataset = CustomDataset(test_encodings)

#Function to get predictions from the model
def get_predictions(model, test_dataset):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    dataloader = DataLoader(test_dataset, batch_size=32)
    device = model.device  # Get the device the model is on

    with torch.no_grad():  # Disable gradient calculation
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}  # Move batch to the correct device
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()  # Move predictions to CPU
            predictions.extend(preds)
    return predictions

#Get predictions on the test set
test_predictions = get_predictions(model, test_dataset)

#Create a DataFrame for the predictions
output_df = pd.DataFrame({"index": test_df['index'], "prediction": test_predictions})

#Save the predictions to a file in the format required for GLUE submission
output_df.to_csv(r"qnli-question-answering/src/qnli_question_answering/results/qnli_test_predictions.tsv", sep='\t', index=False, header=True)

print("Test predictions saved to qnli_test_predictions.tsv")