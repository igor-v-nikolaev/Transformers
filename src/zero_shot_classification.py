from transformers import pipeline

clf = pipeline('zero-shot-classification', device=0)

clf("This is a great movie.", candidate_labels=["positive", "negative"])