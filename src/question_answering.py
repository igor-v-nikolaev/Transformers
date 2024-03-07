from transformers import pipeline

qa = pipeline('question-answering', device=0)

ctx = "Today, I made a peanut butter sandwich"
q = "What did I put in my sandwich?"

qa(question=q, context=ctx)