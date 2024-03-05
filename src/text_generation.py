from transformers import pipeline

gen = pipeline('text-generation')

gen("In this notebook, we will")