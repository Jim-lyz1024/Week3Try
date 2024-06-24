# Load model directly
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Define domain names
# domains = ['mnist', 'mnist_m', 'svhn', 'syn']
# domains = ['art painting', 'cartoon', 'photo', 'sketch']
# domains = ['art', 'clipart', 'product', 'real_world']
# domains = ['caltech', 'labelme','pascal', 'sun']
# domains = ['location_38', 'location_43', 'location_46', 'location_100']
# domains = ['autumn', 'dim', 'grass', 'outdoor', 'rock', 'water']
# domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
domains = ['a picture of a art_paintings dog', 'a picture of a cartoon dog', 'a picture of a photo dog', 'a picture of a sketch dog']

def get_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    return outputs.logits.mean(dim=1).squeeze().detach().numpy()

embeddings = [get_embeddings(domain) for domain in domains]

similarity_matrix = cosine_similarity(embeddings)

similarity_df = pd.DataFrame(similarity_matrix, index=domains, columns=domains)

print(similarity_df)