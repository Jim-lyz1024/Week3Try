import datetime
import os
import time
from collections import OrderedDict

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import DataManager
from evaluator import build_evaluator
from utils import AverageMeter, MetricMeter


class Trainer:
    """Generic Trainer Class for Implementing Generic Function"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR
        self.device = torch.cuda.current_device()

        self._writer = None

        # Build Data Manager
        self.data_manager = DataManager(self.cfg)
        self.data_loader_train = self.data_manager.data_loader_train
        self.data_loader_val = self.data_manager.data_loader_val
        self.data_loader_test = self.data_manager.data_loader_test
        self.num_classes = self.data_manager.num_classes
        self.class_label_name_mapping = self.data_manager.class_label_name_mapping

        self._models = OrderedDict()
        self._optimizers = OrderedDict()
        self._lr_schedulers = OrderedDict()

        # Build Model
        self.build_model()

        # Build Evaluator
        self.evaluator = build_evaluator(
            cfg, class_label_name_mapping=self.class_label_name_mapping
        )

    def build_model(self):
        raise NotImplementedError

    def set_model_mode(self, mode="train", model_names=None):
        # assert mode in ['train', 'eval'], "Invalid mode. Expected 'train' or 'eval', got {}".format(mode)
        # try:
        #     self.model.mode = mode
        # except:
        #     pass
        
        model_names = self.get_model_names(model_names)

        for model_name in model_names:
            if mode == "train":
                self._models[model_name].train()
            elif mode in ["test", "eval"]:
                self._models[model_name].eval()
            else:
                raise KeyError

    def update_lr(self, model_names=None):
        model_names = self.get_model_names(model_names)

        for model_name in model_names:
            if self._lr_schedulers[model_name] is not None:
                self._lr_schedulers[model_name].step()

    def detect_abnormal_loss(self, loss):
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Loss is Infinite or NaN.")

    def model_zero_grad(self, model_names=None):
        model_names = self.get_model_names(model_names)
        for model_name in model_names:
            if self._optimizers[model_name] is not None:
                self._optimizers[model_name].zero_grad()

    def model_backward(self, loss):
        self.detect_abnormal_loss(loss)
        loss.backward()

    def model_update(self, model_names=None):
        model_names = self.get_model_names(model_names)
        for model_name in model_names:
            if self._optimizers[model_name] is not None:
                self._optimizers[model_name].step()

    def model_backward_and_update(self, loss, model_names=None):
        self.model_zero_grad(model_names)
        self.model_backward(loss)
        self.model_update(model_names)

    def init_writer(self, log_dir):
        if self._writer is None:
            print("Initializing Summary Writer with log_dir={}".format(log_dir))
            self._writer = SummaryWriter(log_dir=log_dir)

    def close_writer(self):
        if self._writer is not None:
            self._writer.close()

    def write_scalar(self, tag, scalar_value, global_step=None):
        if self._writer is not None:
            self._writer.add_scalar(tag, scalar_value, global_step)

    def train(self):
        self.before_train()
        for self.current_epoch in range(self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
            # 
            if self.cfg.MODEL.NAME == "CLIPAdapters" or "CLIPAdapter":
                self.evaluate_after_epoch()
        self.after_train()
    
    #####
    def evaluate_after_epoch(self):
            print(f"Evaluating model after epoch {self.current_epoch + 1}")
            accuracy = self.test()  # Assuming 'test' method returns accuracy
            print(f"Accuracy after epoch {self.current_epoch + 1}: {accuracy:.2f}%")
            # Optionally log accuracy to tensorboard or other logging tools
            self.write_scalar('Accuracy/Val', accuracy, self.current_epoch + 1)


    def before_train(self):
        # Initialize SummaryWriter
        # writer_dir = osp.join(self.output_dir, "tensorboard")
        # mkdir_if_missing(writer_dir)
        # self.init_writer(writer_dir)
        self.time_start = time.time()

    def after_train(self):
        print("Finish Training")
        self.extract_source_features_and_plot_tsne()
        self.test()

    def run_epoch(self):
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.data_loader_train)
        end_time = time.time()

        for self.batch_idx, batch_data in enumerate(self.data_loader_train):
            data_time.update(time.time() - end_time)
            loss_summary = self.forward_backward(batch_data)
            batch_time.update(time.time() - end_time)
            losses.update(loss_summary)

            if (
                (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
                or self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            ):
                num_batches_remain = 0
                num_batches_remain += self.num_batches - self.batch_idx - 1
                num_batches_remain += (
                    self.max_epoch - self.current_epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * num_batches_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.current_epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"{losses}"]
                info += [f"lr {self.optimizer.param_groups[0]['lr']:.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            end_time = time.time()

    def before_epoch(self):
        pass

    def after_epoch(self):
        if self.current_epoch + 1 == self.max_epoch:
            self.save_model(self.current_epoch, self.output_dir)

    @torch.no_grad()
    def test(self, split=None):
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT
        if split == "Validation" and self.data_loader_val is not None:
            data_loader = self.data_loader_val
        elif split == "Test":
            data_loader = self.data_loader_test
        else:
            raise NotImplementedError

        print("Evaluate on the {} Set".format(split))

        # Check if get_image_features is implemented
        collect_features = hasattr(self, 'get_image_features')

        if collect_features:
            all_image_features = []
            all_class_labels = []
            all_domain_labels = []

        for _, batch_data in enumerate(tqdm(data_loader)):
            input_data, class_label, domain_label = self.parse_batch_test(batch_data)
            output = self.model_inference(input_data)
            self.evaluator.process(output, class_label)

            if collect_features:
                image_features = self.get_image_features(input_data)
                all_image_features.append(image_features.cpu())
                all_class_labels.append(class_label.cpu())
                all_domain_labels.append(domain_label.cpu())

        evaluation_results = self.evaluator.evaluate()

        if collect_features:
            all_image_features = torch.cat(all_image_features, dim=0)
            all_class_labels = torch.cat(all_class_labels, dim=0)
            all_domain_labels = torch.cat(all_domain_labels, dim=0)

            # Map domain labels to domain names
            domain_labels_np = all_domain_labels.numpy()
            input_domains = self.data_manager.dataset.input_domains
            domain_names = [input_domains[label] for label in domain_labels_np]

            # Map class labels to class names
            class_labels_np = all_class_labels.numpy()
            class_label_name_mapping = self.data_manager.dataset.class_label_name_mapping
            class_names = [class_label_name_mapping[label] for label in class_labels_np]

            # Iterate over each class
            unique_classes = torch.unique(all_class_labels)
            for class_id in unique_classes:
                idx = (all_class_labels == class_id)
                idx_np = idx.cpu().numpy()
                class_image_features = all_image_features[idx]
                # Filter domain names based on idx
                class_domain_names = [domain_names[i] for i in range(len(domain_names)) if idx_np[i]]

                # Perform TSNE
                from sklearn.manifold import TSNE
                image_features_np = class_image_features.numpy()
                tsne = TSNE(n_components=2, random_state=42)
                image_features_2d = tsne.fit_transform(image_features_np)

                # Plot TSNE
                import matplotlib.pyplot as plt
                import seaborn as sns

                plt.figure(figsize=(8, 8))
                sns.scatterplot(
                    x=image_features_2d[:, 0],
                    y=image_features_2d[:, 1],
                    hue=class_domain_names,
                    palette='Set1',
                    legend='full',
                    alpha=0.7
                )
                class_name = class_label_name_mapping[class_id.item()]
                plt.title(f'TSNE of Class {class_name}')
                plt.legend(title='Domain')
                plt.tight_layout()
                # plt.savefig(os.path.join('./', f'tsne_class_{class_id}.png'))
                plt.close()

        return list(evaluation_results.values())[0]

    def parse_batch_train(self, batch_data):
        image = batch_data["img"].to(self.device)
        class_label = batch_data["class_label"].to(self.device)
        domain_label = batch_data["domain_label"].to(self.device)
        
        # domain_label = batch_data["domain_label"].to(self.device)

        # return image, class_label, domain_label
        return image, class_label, domain_label

    def parse_batch_test(self, batch_data):
        input_data = batch_data["img"].to(self.device)
        class_label = batch_data["class_label"].to(self.device)
        domain_label = batch_data["domain_label"].to(self.device)
        return input_data, class_label, domain_label

    def forward_backward(self, batch_data):
        raise NotImplementedError

    def model_inference(self, input_data):
        return self.model(input_data)

    def get_current_lr(self):
        raise NotImplementedError

    def model_registeration(
        self, model_name="model", model=None, optimizer=None, lr_scheduler=None
    ):
        assert model_name not in self._models, "Found duplicate model names."

        self._models[model_name] = model
        self._optimizers[model_name] = optimizer
        self._lr_schedulers[model_name] = lr_scheduler

    def get_model_names(self, model_names=None):
        if model_names is not None:
            if not isinstance(model_names, list):
                model_names = [model_names]

            for model_name in model_names:
                assert model_name in list(self._models.keys())
            return model_names
        else:
            return list(self._models.keys())

    def save_model(
        self,
        current_epoch,
        save_dir,
        model_name="name",
    ):
        model_names = self.get_model_names()

        for model_name in model_names:
            model_dict = self._models[model_name].state_dict()

            optimizer_dict = None
            if self._optimizers[model_name] is not None:
                optimizer_dict = self._optimizers[model_name].state_dict()

            lr_scheduler_state_dict = None
            if self._lr_schedulers[model_name] is not None:
                lr_scheduler_state_dict = self._lr_schedulers[model_name].state_dict()

            # Remove "module." in state_dict's keys
            new_model_dict = OrderedDict()
            for key, value in model_dict.items():
                if key.startswith("module."):
                    key = key[7:]
                new_model_dict[key] = value
            model_dict = new_model_dict

            fpath = os.path.join(save_dir, "model.pth.tar-" + str(current_epoch + 1))
            torch.save(
                {
                    "state_dict": model_dict,
                    "epoch": current_epoch + 1,
                    "optimizer": optimizer_dict,
                    "lr_scheduler": lr_scheduler_state_dict,
                },
                fpath,
            )
            print("Model Saved to: {}".format(fpath))
            
    def get_image_features(self, input_data):
        raise NotImplementedError("get_image_features method not implemented.")

    def extract_source_features_and_plot_tsne(self):
        self.set_model_mode("eval")

        data_loader = self.data_loader_train  # Source domain data

        if data_loader is None:
            print("Source data loader is not available.")
            return

        print("Extracting features from source domains")

        all_image_features = []
        all_class_labels = []
        all_domain_labels = []

        for _, batch_data in enumerate(tqdm(data_loader)):
            input_data, class_label, domain_label = self.parse_batch_train(batch_data)
            image_features = self.get_image_features(input_data)
            all_image_features.append(image_features.cpu())
            all_class_labels.append(class_label.cpu())
            all_domain_labels.append(domain_label.cpu())

        all_image_features = torch.cat(all_image_features, dim=0)
        all_class_labels = torch.cat(all_class_labels, dim=0)
        all_domain_labels = torch.cat(all_domain_labels, dim=0)

        # Map domain labels to domain names
        domain_labels_np = all_domain_labels.numpy()
        input_domains = self.data_manager.dataset.input_domains
        domain_names = [input_domains[label] for label in domain_labels_np]

        # Map class labels to class names
        class_labels_np = all_class_labels.numpy()
        class_label_name_mapping = self.data_manager.dataset.class_label_name_mapping
        class_names = [class_label_name_mapping[label] for label in class_labels_np]

        # Perform UMAP with supervision
        import umap
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=25,  # Decrease to focus on local structure (try values like 5, 10, 15)
            min_dist=0.5,    # Increase to spread out the clusters (try values like 0.1, 0.3, 0.5)
            random_state=42
        )
        embedding = reducer.fit_transform(all_image_features.numpy(), y=class_labels_np)

        # Create a DataFrame for plotting
        import pandas as pd
        df = pd.DataFrame({
            'UMAP1': embedding[:, 0],
            'UMAP2': embedding[:, 1],
            'Domain': domain_names,
            'Class': class_names
        })

        # Plot using seaborn
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(12, 10))
        sns.scatterplot(
            data=df,
            x='UMAP1', y='UMAP2',
            hue='Domain',
            palette='Set1',
            marker='o',  # Use the same marker for all points
            s=60,
            alpha=0.7
        )

        # Annotate clusters with class names
        # Compute the centroid of each class cluster
        centroids = df.groupby('Class')[['UMAP1', 'UMAP2']].mean().reset_index()
        for _, row in centroids.iterrows():
            plt.text(row['UMAP1'], row['UMAP2'], row['Class'], fontsize=12, fontweight='bold')

        plt.title('UMAP of Source Domain Image Features')
        plt.legend(title='Domain', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'umap_source_domains.png'))
        plt.close()
