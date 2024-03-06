from transformers import pipeline

mlm = pipeline('fill-mask', model='bert-base-uncased')

mlm("The quick brown <mask> jumps over the lazy dog")
mlm("The [MASK] jumps over the lazy dog")