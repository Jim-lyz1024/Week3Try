import os
import numpy as np
from scipy.special import rel_entr

def calculate_distribution(data_dir, domain):
    distribution = {}
    for split in ['train', 'val']:
        split_dir = os.path.join(data_dir, domain, split)
        if not os.path.exists(split_dir):
            continue

        classes = os.listdir(split_dir)
        for cls in classes:
            class_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(class_dir):
                continue

            num_files = len(os.listdir(class_dir))
            if cls in distribution:
                distribution[cls] += num_files
            else:
                distribution[cls] = num_files

    total_files = sum(distribution.values())
    for cls in distribution:
        distribution[cls] /= total_files
    
    return distribution

def calculate_kl_divergence(dist1, dist2):
    keys = set(dist1.keys()).union(set(dist2.keys()))
    p = np.array([dist1.get(k, 0.0) for k in keys])
    q = np.array([dist2.get(k, 0.0) for k in keys])
    
    # To avoid division by zero
    p = np.where(p == 0, 1e-10, p)
    q = np.where(q == 0, 1e-10, q)
    
    return np.sum(rel_entr(p, q))

source_dir = "../data/office_home"
target_dir = "../data/office_home_adjustied_processed"
# target_dir = "data/vlcs"

domains = ["art", "clipart", "product", "real_world"]

for domain in domains:
    source_distribution = calculate_distribution(source_dir, domain)
    target_distribution = calculate_distribution(target_dir, domain)
    
    kl_divergence = calculate_kl_divergence(source_distribution, target_distribution)
    print(f"KL divergence between original and adjusted {domain} domain: {kl_divergence:.4f}")
