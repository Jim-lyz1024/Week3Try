import os
import random
from collections import defaultdict

from datasets.base_dataset import DatasetBase, Datum
from datasets.build_dataset import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class PACSSubdomain(DatasetBase):
    """
    PACS-Subdomain Dataset:
        - 9 source domains: art_painting0, art_painting1, art_painting2, cartoon0, cartoon1, cartoon2, photo0, photo1, photo2
        - 1 target domain: sketch
        - 7 categories: dog, elephant, giraffe, guitar, horse, house and person.
    """

    def __init__(self, cfg):
        self._dataset_dir = "PACS-subdomain"
        self._domains = [
            "art_painting0", "art_painting1", "art_painting2",
            "cartoon0", "cartoon1", "cartoon2",
            "photo0", "photo1", "photo2",
            "sketch"
        ]
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self._dataset_dir = os.path.join(root, self._dataset_dir)
        self._error_img_paths = ["sketch/dog/n02103406_4068-1.png"]

        self.check_input_domains(cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS)

        train_data, val_data = self.read_data(cfg.DATASET.SOURCE_DOMAINS, split_ratio=0.9)
        test_data = self.read_data(cfg.DATASET.TARGET_DOMAINS, split_ratio=1.0)  # Use all target domain data for testing
        
        super().__init__(
            dataset_dir=self._dataset_dir,
            domains=self._domains,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
        )

    def read_data(self, input_domains, split_ratio=0.9):
        data = []
        for domain_label, domain_name in enumerate(input_domains):
            domain_dir = os.path.join(self._dataset_dir, domain_name)
            for class_name in os.listdir(domain_dir):
                class_dir = os.path.join(domain_dir, class_name)
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
        
        if split_ratio < 1.0:
            random.shuffle(data)
            split_index = int(len(data) * split_ratio)
            return data[:split_index], data[split_index:]
        else:
            return data

    def _get_class_label(self, class_name):
        class_names = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
        return class_names.index(class_name.lower())

    @property
    def class_names(self):
        return ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]

    @property
    def domains(self):
        return self._domains