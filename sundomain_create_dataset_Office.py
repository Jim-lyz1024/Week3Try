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
        for split in ['train', 'val']:
            subdomain_dir = os.path.join(new_dataset_path, f"{domain_name}{subdomain}", split)
            os.makedirs(subdomain_dir, exist_ok=True)
    
    # Copy images to new structure
    for relative_path, subdomain in tqdm(subdomain_mapping.items(), desc=f"Copying images for {domain_name}"):
        src_path = os.path.join(original_dataset_path, relative_path)
        split = os.path.basename(os.path.dirname(os.path.dirname(relative_path)))  # 'train' or 'val'
        class_name = os.path.basename(os.path.dirname(relative_path))
        dst_dir = os.path.join(new_dataset_path, f"{domain_name}{subdomain}", split, class_name)
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, os.path.basename(relative_path))
        shutil.copy2(src_path, dst_path)

    print(f"Created subdomain dataset for {domain_name} at {new_dataset_path}")

# Usage
original_dataset_path = "data/office_home"
new_dataset_path = "data/OfficeHome-subdomain-clipart"
# new_dataset_path = "data/OfficeHome-subdomain-DB-clipart"
# domains = ["clipart", "product", "real_world"]
domains = ["art"]

for domain in domains:
    create_subdomain_dataset(
        os.path.join(original_dataset_path, domain),
        f"subdomain_{domain}_mapping.json",
        # f"subdomain_DBSCAN_{domain}_mapping.json",
        new_dataset_path,
        domain
    )

print("Subdomain dataset creation completed.")