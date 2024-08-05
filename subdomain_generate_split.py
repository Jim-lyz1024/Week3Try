import os
import random
from glob import glob

def generate_subdomain_splits(root_dir, output_dir):
    domains = ["art_painting0", "art_painting1", "art_painting2",
               "cartoon0", "cartoon1", "cartoon2",
               "photo0", "photo1", "photo2",
               "sketch"]
    
    os.makedirs(output_dir, exist_ok=True)
    
    for domain in domains:
        all_images = []
        for class_name in os.listdir(os.path.join(root_dir, domain)):
            class_dir = os.path.join(root_dir, domain, class_name)
            if os.path.isdir(class_dir):
                images = glob(os.path.join(class_dir, "*.jpg")) + glob(os.path.join(class_dir, "*.png"))
                all_images.extend([(img, class_name) for img in images])
        
        random.shuffle(all_images)
        
        # Split ratios (you can adjust these as needed)
        train_ratio, val_ratio = 0.7, 0.15
        
        train_split = all_images[:int(len(all_images) * train_ratio)]
        val_split = all_images[int(len(all_images) * train_ratio):int(len(all_images) * (train_ratio + val_ratio))]
        test_split = all_images[int(len(all_images) * (train_ratio + val_ratio)):]
        
        # Write split files
        for split_name, split_data in [("train", train_split), ("crossval", val_split), ("test", test_split)]:
            with open(os.path.join(output_dir, f"{domain}_{split_name}_kfold.txt"), "w") as f:
                for img_path, class_name in split_data:
                    relative_path = os.path.relpath(img_path, root_dir)
                    class_label = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"].index(class_name) + 1
                    f.write(f"{relative_path} {class_label}\n")

if __name__ == "__main__":
    root_dir = "data/PACS-subdomain"
    output_dir = "data/PACS-subdomain/splits"
    generate_subdomain_splits(root_dir, output_dir)
    print("Split files generated successfully.")