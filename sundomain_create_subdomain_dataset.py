import os
import json
import shutil
from tqdm import tqdm

def create_subdomain_dataset(original_dataset_path, subdomain_mapping_path, new_dataset_path, domain_name):
    # Load subdomain mapping
    with open(subdomain_mapping_path, 'r') as f:
        subdomain_mapping = json.load(f)
    
    # Create new dataset directory
    os.makedirs(new_dataset_path, exist_ok=True)
    
    # Create subdomain directories
    subdomains = set(subdomain_mapping.values())
    for subdomain in subdomains:
        subdomain_dir = os.path.join(new_dataset_path, f"{domain_name}{subdomain}")
        os.makedirs(subdomain_dir, exist_ok=True)
    
    # Copy images to new structure
    for relative_path, subdomain in tqdm(subdomain_mapping.items(), desc="Copying images"):
        src_path = os.path.join(original_dataset_path, relative_path)
        dst_dir = os.path.join(new_dataset_path, f"{domain_name}{subdomain}", os.path.dirname(relative_path))
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, os.path.basename(relative_path))
        shutil.copy2(src_path, dst_path)

    print(f"Created subdomain dataset for {domain_name} at {new_dataset_path}")

# Usage
original_dataset_path = "data/pacs_copy/images"
subdomain_mapping_path = "subdomain_mapping.json"
new_dataset_path = "data/PACS-subdomain"
domains = ["art_painting", "cartoon", "photo"]

for domain in domains:
    create_subdomain_dataset(
        os.path.join(original_dataset_path, domain),
        f"subdomain_{domain}_mapping.json",
        new_dataset_path,
        domain
    )

print("Subdomain dataset creation completed.")