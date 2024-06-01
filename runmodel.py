from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import pandas as pd
from datasets import Dataset


model_name = "VitaRin/ChemBERTaNLRP3"
smiles_data = ""

pipeline = TextClassificationPipeline(
    model=AutoModelForSequenceClassification.from_pretrained(model_name),
    tokenizer=AutoTokenizer.from_pretrained(model_name),
    device=0
)

test_df = pd.read_csv(smiles_data)
test_dataset = Dataset.from_pandas(test_df)
molecules = list(test_dataset["text"])

result = pipeline(molecules)

print(result)