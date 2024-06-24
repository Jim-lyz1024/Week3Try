import torch
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')

domains = ['art', 'clipart', 'product', 'real_world']

def get_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

embeddings = [get_embeddings(domain) for domain in domains]

similarity_matrix = cosine_similarity(embeddings)

similarity_df = pd.DataFrame(similarity_matrix, index=domains, columns=domains)

print(similarity_df)
