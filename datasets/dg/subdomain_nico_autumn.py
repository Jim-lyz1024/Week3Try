import os
import random
from collections import defaultdict

from datasets.base_dataset import DatasetBase, Datum
from datasets.build_dataset import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class NICOSubdomainAutumn(DatasetBase):
    """
    NICO-subdomain-autumn Dataset:
        - 10 source domains: autumn0, autumn1, autumn2, autumn3, autumn4, autumn5, dim, grass, outdoor, rock
        - 1 target domain: water
        - 79 categories of objects in autumn scenes.
    """

    def __init__(self, cfg):
        self._dataset_dir = "NICO-subdomain-autumn"
        self._domains = [
            "autumn0", "autumn1", "autumn2", "autumn3", "autumn4", "autumn5", 
            "dim", "grass", "outdoor", "rock", "water"
        ]
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self._dataset_dir = os.path.join(root, self._dataset_dir)

        self.check_input_domains(cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS)

        # Initialize class names from all domains
        self._class_names = self._get_class_names(cfg.DATASET.SOURCE_DOMAINS + cfg.DATASET.TARGET_DOMAINS)

        # Read data
        train_data, val_data = self.read_data(cfg.DATASET.SOURCE_DOMAINS, split="train")
        test_data = self.read_data(cfg.DATASET.TARGET_DOMAINS, split="test")
        
        super().__init__(
            dataset_dir=self._dataset_dir,
            domains=self._domains,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
        )

    def read_data(self, input_domains, split):
        data = []
        val_data = []  # List to store validation data

        for domain_label, domain_name in enumerate(input_domains):
            domain_dir = os.path.join(self._dataset_dir, domain_name)

            for class_name in os.listdir(domain_dir):
                class_dir = os.path.join(domain_dir, class_name)
                if not os.path.isdir(class_dir) or class_name == "val":
                    continue

                class_label = self._get_class_label(class_name)
                class_images = [os.path.join(class_dir, img_name) for img_name in os.listdir(class_dir)]

                if domain_name == "dim":  # Assume 'water' is the target domain
                    if split == "test":
                        data.extend([
                            Datum(
                                img_path=img_path,
                                class_label=class_label,
                                domain_label=domain_label,
                                class_name=class_name,
                            ) for img_path in class_images
                        ])
                else:
                    if split == "train":
                        train_split_index = int(len(class_images) * 0.8)
                        train_images = class_images[:train_split_index]
                        val_images = class_images[train_split_index:]

                        data.extend([
                            Datum(
                                img_path=img_path,
                                class_label=class_label,
                                domain_label=domain_label,
                                class_name=class_name,
                            ) for img_path in train_images
                        ])

                        val_data.extend([
                            Datum(
                                img_path=img_path,
                                class_label=class_label,
                                domain_label=domain_label,
                                class_name=class_name,
                            ) for img_path in val_images
                        ])
                    elif split == "val":
                        val_data.extend([
                            Datum(
                                img_path=img_path,
                                class_label=class_label,
                                domain_label=domain_label,
                                class_name=class_name,
                            ) for img_path in class_images
                        ])

        # When split is 'train', return both train and val data; otherwise, just return data
        if split == "train":
            return data, val_data
        else:
            return data

    def _get_class_names(self, domains):
        class_names = set()
        for domain in domains:
            domain_dir = os.path.join(self._dataset_dir, domain)
            class_names.update([d.lower() for d in os.listdir(domain_dir) if os.path.isdir(os.path.join(domain_dir, d)) and d != "val"])
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
