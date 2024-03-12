from transformers import AutoTokenizer

checkpoint = "bert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

tokenizer("Hello World")

tokens = tokenizer.tokenize("Hello World")
ids0 = tokenizer.convert_tokens_to_ids(tokens)
ids = tokenizer.encode("Hello World")

tokenizer.convert_ids_to_tokens(ids)
tokenizer.decode(ids)

tokenizer("Hello World", return_tensors="pt")
tokenizer("Hello World", return_tensors="tf")
tokenizer("Hello World", return_tensors="np")

data = ["I like cats", "Do you like cats too?"]

model_inputs = tokenizer(data, padding=True, truncation=True, return_tensors="pt")

