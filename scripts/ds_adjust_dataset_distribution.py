import os
import random
from shutil import copyfile

def adjust_dataset_distribution(source_dir, target_dir, domain, shift_factor):
    # 遍历train和val目录
    for split in ['train', 'val']:
        split_source_dir = os.path.join(source_dir, domain, split)
        split_target_dir = os.path.join(target_dir, domain, split)

        if not os.path.exists(split_target_dir):
            os.makedirs(split_target_dir)

        # 遍历每个class目录
        classes = os.listdir(split_source_dir)
        for cls in classes:
            class_source_dir = os.path.join(split_source_dir, cls)
            class_target_dir = os.path.join(split_target_dir, cls)

            if not os.path.exists(class_target_dir):
                os.makedirs(class_target_dir)

            files = os.listdir(class_source_dir)
            random.shuffle(files)

            num_files = len(files)
            num_to_select = int(num_files * shift_factor)

            selected_files = files[:num_to_select]

            for file in selected_files:
                src_file = os.path.join(class_source_dir, file)
                dst_file = os.path.join(class_target_dir, file)
                copyfile(src_file, dst_file)
            
            print(f"Selected {num_to_select} files from {cls} in {domain}/{split}.")

source_dir = "data/office_home"
target_dir = "data/office_home_adjusted"
domains = ["art", "clipart", "product", "real_world"]
shift_factors = [0.3, 0.4, 0.5, 0.4]  # Adjust these values as needed

for domain, shift_factor in zip(domains, shift_factors):
    adjust_dataset_distribution(source_dir, target_dir, domain, shift_factor)
