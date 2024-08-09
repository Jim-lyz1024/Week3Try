import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from scipy.stats import entropy
from glob import glob
import os
import logging
import json

logging.basicConfig(level=logging.INFO)

# Load pre-trained ResNet model
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image_paths):
    features = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                feature = model(img_tensor).squeeze().numpy()
            features.append(feature)
        except Exception as e:
            logging.error(f"Error processing {img_path}: {str(e)}")
    return np.array(features)

# Adjust the base path
script_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.abspath(os.path.join(script_dir, '..', 'data', 'office_home'))

print(f"Base path: {base_path}")
if not os.path.exists(base_path):
    raise FileNotFoundError(f"The directory {base_path} does not exist.")

# Process source domains (clipart, product, real_world)
source_domains = ['clipart', 'product', 'real_world']
for domain in source_domains:
    domain_path = os.path.join(base_path, domain)
    image_paths = []
    
    for split in ['train', 'val']:
        split_path = os.path.join(domain_path, split)
        for class_dir in os.listdir(split_path):
            class_path = os.path.join(split_path, class_dir)
            if os.path.isdir(class_path):
                class_images = glob(os.path.join(class_path, '*.jpg'))
                image_paths.extend(class_images)
                print(f"Found {len(class_images)} images in {domain}/{split}/{class_dir}")

    if not image_paths:
        raise FileNotFoundError(f"No images found in {domain_path}")

    logging.info(f"Found {len(image_paths)} images in total for {domain}")

    # Extract features
    logging.info(f"Extracting features for {domain}...")
    features = extract_features(image_paths)

    if features.shape[0] == 0:
        raise ValueError(f"No features were extracted for {domain}. Check if the images are valid.")

    logging.info(f"Extracted features shape for {domain}: {features.shape}")

    # Adjust n_components and perplexity based on sample size
    n_components = min(2, features.shape[1] - 1)
    perplexity = min(30, features.shape[0] - 1)

    logging.info(f"Using n_components={n_components} and perplexity={perplexity} for t-SNE")

    # Dimensionality reduction
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    features_2d = tsne.fit_transform(features)

    # Clustering
    n_clusters = min(3, features.shape[0])  # Adjust based on your needs
    logging.info(f"Using n_clusters={n_clusters} for KMeans")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features_2d)

    # Calculate KL divergence between clusters
    def kl_divergence(p, q):
        return entropy(p, q)

    kl_matrix = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            if i != j:
                p = features[cluster_labels == i].mean(axis=0)
                q = features[cluster_labels == j].mean(axis=0)
                kl_matrix[i, j] = kl_divergence(p, q)

    logging.info(f"KL divergence matrix for {domain}:")
    logging.info(kl_matrix)

    # Assign subdomains
    subdomains = [f"subdomain_{label}" for label in cluster_labels]

    # Print subdomain distribution
    subdomain_counts = np.bincount(cluster_labels)
    for i, count in enumerate(subdomain_counts):
        logging.info(f"Subdomain {i} in {domain}: {count} images")

    # Create subdomain mapping
    subdomain_mapping = {}
    for i, (image_path, label) in enumerate(zip(image_paths, cluster_labels)):
        relative_path = os.path.relpath(image_path, domain_path)
        subdomain_mapping[relative_path] = int(label)

    # Save the subdomain mapping to a JSON file
    with open(f'subdomain_{domain}_mapping.json', 'w') as f:
        json.dump(subdomain_mapping, f)

    logging.info(f"Subdomain mapping for {domain} saved to subdomain_{domain}_mapping.json")

logging.info("Subdomain division completed for all source domains.")