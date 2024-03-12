from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

data = ["I like cats", "Do you like cats too?"]

model_inputs = tokenizer(data, padding=True, truncation=True, return_tensors="pt")

outputs = model(**model_inputs)