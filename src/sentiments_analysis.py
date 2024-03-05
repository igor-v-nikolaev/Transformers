import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from transformers import pipeline

# Load the sentiment analysis pipeline
classifier = pipeline('sentiment-analysis')

classifier('We are very happy to show you the 🤗 Transformers library.')