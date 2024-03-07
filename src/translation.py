from transformers import pipeline

# huggingface.co/models
translator = pipeline('translation', model='Helsinki-NLP/opus-mt-en-de', device=0)

translator("I like eggs and ham.")