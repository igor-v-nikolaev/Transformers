from transformers import pipeline

ner = pipeline('ner', aggregation_strategy='simple', device=0)

ner("Hugging Face is a company based in New York City. Its headquarters are in DUMBO, therefore very close to the Manhattan Bridge.")