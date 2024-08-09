import os

def count_samples(dataset_path):
    total_count = 0
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                total_count += 1
    return total_count

original_path = "data/office_home"
original_count = count_samples(original_path)
print(f"Original dataset sample count: {original_count}")

subdomain_path = "data/OfficeHome-subdomain"
subdomain_count = count_samples(subdomain_path)
print(f"Subdomain dataset sample count: {subdomain_count}")