import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import entropy
from glob import glob
import os
import logging
import json
from itertools import product
from scipy.special import kl_div

logging.basicConfig(level=logging.INFO)

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

def extract_and_save_features(image_paths, feature_file):
    if os.path.exists(feature_file):
        logging.info(f"Loading features from {feature_file}")
        return np.load(feature_file)
    
    logging.info("Extracting features...")
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
    
    features = np.array(features)
    np.save(feature_file, features)
    logging.info(f"Features saved to {feature_file}")
    return features

def kl_divergence(p, q):
    p = np.clip(p, 1e-10, 1)
    q = np.clip(q, 1e-10, 1)

    p /= np.sum(p)
    q /= np.sum(q)
    return np.sum(kl_div(p, q))

def calculate_kl_divergence(features, labels):
    unique_labels = sorted(list(set(labels) - {-1})) 
    n_clusters = len(unique_labels)
    kl_matrix = np.zeros((n_clusters, n_clusters))
    for i, label_i in enumerate(unique_labels):
        for j, label_j in enumerate(unique_labels):
            if i != j:
                p = features[labels == label_i].mean(axis=0)
                q = features[labels == label_j].mean(axis=0)
                kl_matrix[i, j] = kl_divergence(p, q)
    avg_kl = np.mean(kl_matrix)
    return avg_kl, kl_matrix

def try_clustering_parameters(features, eps_values, min_samples_values):
    results = []
    for eps, min_samples in product(eps_values, min_samples_values):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = dbscan.fit_predict(features)
        n_clusters = len(set(labels) - {-1})  
        logging.info(f"DBSCAN (Cosine) with eps={eps}, min_samples={min_samples} produced {n_clusters} clusters")
        if 1 < n_clusters <= 10:
            result = process_clustering_result(features, labels, f"DBSCAN (Cosine), eps={eps}, min_samples={min_samples}")
            results.append(result)
            logging.info(f"Average KL divergence: {result['avg_kl']}")
    return results

def process_clustering_result(features, labels, method):
    unique_labels = sorted(list(set(labels) - {-1})) 
    n_clusters = len(unique_labels)
    avg_kl, kl_matrix = calculate_kl_divergence(features, labels)
    return {
        'method': method,
        'n_clusters': n_clusters,
        'avg_kl': float(avg_kl),
        'kl_matrix': kl_matrix.tolist(),
        'labels': labels.tolist()
    }

# Adjust the base path for DomainNet
script_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.abspath(os.path.join(script_dir, '..', 'data', 'domainnet'))

print(f"Base path: {base_path}")
if not os.path.exists(base_path):
    raise FileNotFoundError(f"The directory {base_path} does not exist.")

# Process all domains in DomainNet
domains = ['real', 'sketch']
# domains = ['clipart']

for domain in domains:
    domain_path = os.path.join(base_path, domain)
    image_paths = []
    
    for class_dir in os.listdir(domain_path):
        class_path = os.path.join(domain_path, class_dir)
        if os.path.isdir(class_path):
            class_images = glob(os.path.join(class_path, '*.jpg'))  # Assuming jpg format, adjust if needed
            image_paths.extend(class_images)
            print(f"Found {len(class_images)} images in {domain}/{class_dir}")

    if not image_paths:
        raise FileNotFoundError(f"No images found in {domain_path}")

    logging.info(f"Found {len(image_paths)} images in total for {domain}")

    # Extract or load features
    feature_file = f'features_domainnet_{domain}.npy'
    features = extract_and_save_features(image_paths, feature_file)

    logging.info(f"Feature shape for {domain}: {features.shape}")

    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)

    # Apply PCA
    pca = PCA(n_components=0.95)  # Keep 95% of variance
    features_pca = pca.fit_transform(features_normalized)
    logging.info(f"PCA reduced features shape: {features_pca.shape}")

    # Try different clustering parameters
    eps_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    min_samples_values = [2, 5, 10, 20, 30]

    results = try_clustering_parameters(features_pca, eps_values, min_samples_values)

    if not results:
        logging.warning("No parameter combination produced 2-10 clusters. Saving all results.")
        # If no good results, save all results including those with 1 or >10 clusters
        results = try_clustering_parameters(features_normalized, eps_values, min_samples_values)

    # Sort results by average KL divergence (descending order)
    results.sort(key=lambda x: x['avg_kl'], reverse=True)

    # Save results to a JSON file
    with open(f'dbscan_results_domainnet_{domain}.json', 'w') as f:
        json.dump(results, f, indent=2)

    logging.info(f"DBSCAN results for {domain} saved to dbscan_results_domainnet_{domain}.json")

    if results:
        best_result = results[0]
        best_labels = best_result['labels']

        subdomain_mapping = {}
        for i, (image_path, label) in enumerate(zip(image_paths, best_labels)):
            relative_path = os.path.relpath(image_path, domain_path)
            subdomain_mapping[relative_path] = int(label)

        # Save subdomain mapping to a JSON file
        with open(f'subdomain_DBSCAN_domainnet_{domain}_mapping.json', 'w') as f:
            json.dump(subdomain_mapping, f)

        logging.info(f"Subdomain mapping for {domain} saved to subdomain_DBSCAN_domainnet_{domain}_mapping.json")
        logging.info(f"Best method: {best_result['method']}")
        logging.info(f"Number of clusters: {best_result['n_clusters']}")
        logging.info(f"Average KL divergence: {best_result['avg_kl']}")
        logging.info(f"KL divergence matrix: {best_result['kl_matrix']}")
    else:
        logging.warning(f"No suitable clustering found for {domain}. Check the dbscan_results_domainnet_{domain}.json file for details.")

logging.info("Subdomain division completed for all domains in DomainNet.")