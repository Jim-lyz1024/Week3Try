import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Define domain names
# domains = ['mnist', 'mnist_m', 'svhn', 'syn']
domains = ['art painting', 'cartoon', 'photo', 'sketch']
# domains = ['art', 'clipart', 'product', 'real_world']
# domains = ['caltech', 'labelme','pascal', 'sun']
# domains = ['location_38', 'location_43', 'location_46', 'location_100']
# domains = ['water', 'autumn', 'dim', 'grass', 'outdoor', 'rock']
# domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
# domains = ['a picture of a art_paintings dog', 'a picture of a cartoon dog', 'a picture of a photo dog', 'a picture of a sketch dog']


# Function to get text embeddings
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    # Take the mean of the token embeddings to get a single vector for the text
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

# Compute embeddings for each domain name
embeddings = [get_embeddings(domain) for domain in domains]

# Compute cosine similarity matrix
similarity_matrix = cosine_similarity(embeddings)

# Create a DataFrame to display the similarity matrix
similarity_df = pd.DataFrame(similarity_matrix, index=domains, columns=domains)

# Display the similarity matrix
print(similarity_df)
