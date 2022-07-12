from transformers import MPNetTokenizer, MPNetModel
import torch

tokenizer = MPNetTokenizer.from_pretrained("microsoft/mpnet-base")
model = MPNetModel.from_pretrained("microsoft/mpnet-base")

inputs = tokenizer("我是草神纳西妲的猫", return_tensors="pt")
print(inputs)
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state