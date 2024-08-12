import os
import random
from collections import defaultdict

from datasets.base_dataset import DatasetBase, Datum
from datasets.build_dataset import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class OfficeHomeSubdomainClipart(DatasetBase):
    """
    OfficeHome-Subdomain Dataset:
        - 9 source domains: clipart0, clipart1, clipart2, product0, product1, product2, real_world0, real_world1, real_world2
        - 1 target domain: art
        - 65 categories related to office and home objects.
    """

    def __init__(self, cfg):
        self._dataset_dir = "OfficeHome-subdomain-clipart"
        self._domains = [
            "art0", "art1", "art2",
            "product0", "product1", "product2",
            "real_world0", "real_world1", "real_world2", "clipart"
        ]
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self._dataset_dir = os.path.join(root, self._dataset_dir)

        self.check_input_domains(cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS)

        # Initialize class names from all domains
        self._class_names = self._get_class_names(cfg.DATASET.SOURCE_DOMAINS + cfg.DATASET.TARGET_DOMAINS)

        train_data, val_data = self.read_data(cfg.DATASET.SOURCE_DOMAINS, split="train")
        test_data = self.read_data(cfg.DATASET.TARGET_DOMAINS, split="all")
        
        super().__init__(
            dataset_dir=self._dataset_dir,
            domains=self._domains,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
        )

    def read_data(self, input_domains, split):
        data = []
        for domain_label, domain_name in enumerate(input_domains):
            domain_dir = os.path.join(self._dataset_dir, domain_name)
            if split == "all":
                splits_to_use = ["train", "val"]
            else:
                splits_to_use = [split]
            
            for split_name in splits_to_use:
                split_dir = os.path.join(domain_dir, split_name)
                for class_name in os.listdir(split_dir):
                    class_dir = os.path.join(split_dir, class_name)
                    if not os.path.isdir(class_dir):
                        continue
                    class_label = self._get_class_label(class_name)
                    for img_name in os.listdir(class_dir):
                        img_path = os.path.join(class_dir, img_name)
                        data.append(Datum(
                            img_path=img_path,
                            class_label=class_label,
                            domain_label=domain_label,
                            class_name=class_name,
                        ))
        
        if split == "train":
            random.shuffle(data)
            split_index = int(len(data) * 0.9)  # 90% for training, 10% for validation
            return data[:split_index], data[split_index:]
        else:
            return data

    def _get_class_names(self, domains):
        class_names = set()
        for domain in domains:
            domain_dir = os.path.join(self._dataset_dir, domain)
            for split in ['train', 'val']:
                split_dir = os.path.join(domain_dir, split)
                if os.path.exists(split_dir):
                    class_names.update([d.lower() for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
        return sorted(list(class_names))

    def _get_class_label(self, class_name):
        class_name = class_name.lower()
        if class_name not in self._class_names:
            print(f"Warning: Class '{class_name}' not found in class list. Adding it.")
            self._class_names.append(class_name)
        return self._class_names.index(class_name)

    @property
    def class_names(self):
        return self._class_names

    @property
    def domains(self):
        return self._domains