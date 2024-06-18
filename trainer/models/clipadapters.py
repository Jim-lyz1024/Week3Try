import os

import torch
import torch.nn as nn
from clip import clip
from torch.nn import functional as F

from metrics import compute_accuracy
from optim import build_lr_scheduler, build_optimizer
from trainer import MODEL_REGISTRY, Trainer
from utils import PROMPT_TEMPLATES


class Adapter(nn.Module):
    def __init__(self, channel_in, reduction=4): # reduction=4 refer to the original paper
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel_in, channel_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel_in // reduction, channel_in, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class CustomCLIP(nn.Module):
    def __init__(self, cfg, class_names, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        # Adapter for RN50 CLIP Backbone
        # self.adapter = Adapter(1024, 4).to(clip_model.dtype)
        # Adapter for VITB32 CLIP Backbone
        # 512 and 1024 are the default dimensions, using different values for the backbone parameter.
        # self.adapter = Adapter(512, 4).to(clip_model.dtype) 
        # 512 for ViTB32
        self.adapters = nn.ModuleList([Adapter(512, 4).to(clip_model.dtype) for i in range(len(cfg.DATASET.SOURCE_DOMAINS))]) 
        # self.adapters = nn.ModuleList([Adapter(1024, 4).to(clip_model.dtype) for i in range(len(cfg.DATASET.SOURCE_DOMAINS))])
        self.dtype = clip_model.dtype
        self.cfg = cfg
        self.class_names = class_names
        self.clip_model = clip_model
        self.text_features = {}
        self.text_features2 = {}
        self.sim_scores = []
        self.initialize_text_features()
        
    def initialize_text_features(self):
        def generate_prompts(domain_names, prompt_template):
            prompts_domain = {}
            prompts_original = [prompt_template.format(class_name.replace("_", " ")) for class_name in self.class_names]
            prompts_domain['original'] = prompts_original

            for domain in domain_names:
                prompts_domain[domain] = [
                    prompt_template.format(domain.replace("_", " ") + ' ' + class_name.replace("_", " "))
                    for class_name in self.class_names
                ]
            return prompts_domain

        prompt_template = PROMPT_TEMPLATES[self.cfg.DATASET.NAME]
        
        source_domain_names = self.cfg.DATASET.SOURCE_DOMAINS
        target_domain_names = self.cfg.DATASET.TARGET_DOMAINS

        # Generate prompts for source and target domains
        source_prompts = generate_prompts(source_domain_names, prompt_template)
        target_prompts = generate_prompts(target_domain_names, prompt_template)

        print(source_prompts)
        print(target_prompts)

        def encode_text(prompts):
            text_features = {}
            for domain, prompts_list in prompts.items():
                tokenized_prompts = [clip.tokenize(prompt) for prompt in prompts_list]
                tokenized_prompts = torch.cat(tokenized_prompts).to(torch.cuda.current_device())
                with torch.no_grad():
                    text_features[domain] = self.clip_model.encode_text(tokenized_prompts)
                    text_features[domain] = text_features[domain] / text_features[domain].norm(dim=-1, keepdim=True)
            return text_features

        self.text_features = encode_text(source_prompts)
        self.text_features2 = encode_text(target_prompts)

        tar_f = self.text_features2[target_domain_names[0]]
        sim_scores = [F.cosine_similarity(v.flatten(), tar_f.flatten(), dim=0) for v in self.text_features.values()]
        self.sim_scores = sim_scores[1:]
        
        
    def forward(self, image, domain_label=None):
        adapter_ratio = 0.2
        # computes the image features using the CLIP image encoder
        image_features = self.image_encoder(image.type(self.dtype)) # Image Features Shape: torch.Size([64, 512])
        # obtain adapted features
        
        adapter_features = []
        
        if domain_label is not None:
            for itj, d in enumerate(domain_label):
                adapter_features.append(self.adapters[d](image_features[itj:itj+1])) # image_features: torch.Size([1, 512])
            adapter_features = torch.vstack(adapter_features)
        else:
            for adapter in self.adapters:
                adapter_features.append(adapter(image_features))
            # Compute weights using softmax
            weights = F.softmax(torch.tensor(self.sim_scores), dim=0).to(self.dtype)
            combined_adapter_features = sum(w * f for w, f in zip(weights, adapter_features))
            adapter_features = combined_adapter_features

        image_features = ( adapter_ratio * adapter_features + (1 - adapter_ratio) * image_features)
        # image_featuress = [( adapter_ratio * adapter_featuress[i] + (1 - adapter_ratio) * image_features) for i in range(len(self.adapters))]

        # regularization, avoid updating gradient too fast
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # image_featuress = [image_featuress[i] / image_featuress[i].norm(dim=-1, keepdim=True) for i in range(len(self.adapters))]
        
        
        logit_scale = self.logit_scale.exp()
        
        # Calculate logits for each domain
        # logits = logit_scale * image_features @ self.text_features.t() # .t() means transpose of a matrix
        
        ##### In test: only use original prompt
        if domain_label == None:
            logits = logit_scale * image_features @ self.text_features['original'].t()
            return logits
        
        ##### In train: use all prompts
        logits_domain = {}
        # logits_domains = [{} for i in range(len(self.adapters))]
        for ith,(domain, text_feature) in enumerate(self.text_features.items()):
            # Image features shape: torch.Size([64, 512])
            # text_feature.shape = torch.Size([7, 512])
            logits_domain[domain] = logit_scale * image_features @ text_feature.t()
            # for i in range(len(self.adapters)):
            #     logits_domains[i][domain] = logit_scale * image_featuress[i] @ text_feature.t()

        all_domains = torch.cat(list(logits_domain.values()), dim=1) # All Domains: torch.Size([64, 28])
        # all_domainss = [torch.cat(list(logits_domains[i].values()), dim=1) for i in range(len(self.adapters))]

        # cosine_similarity = {}
        # logits_domains = [{} for i in range(len(self.adapters))]
        # for ith,(domain, text_feature) in enumerate(self.text_features.items()):
        #
        #
        #
        #     cosine_similarity[domain] = F.cosine_similarity(image_features, text_feature.t(), dim=1)
        #     for i in range(len(self.adapters)):
        #         logits_domains[i][domain] = logit_scale * image_featuress[i] @ text_feature.t()
        #
        # all_domains = torch.cat(list(logits_domain.values()), dim=1) # All Domains: torch.Size([64, 28])
        # all_domainss = [torch.cat(list(logits_domains[i].values()), dim=1) for i in range(len(self.adapters))]
        
        return all_domains
    
 
@MODEL_REGISTRY.register()
class CLIPAdapters(Trainer):
    """CLIP-Adapter

    CLIP-Adapter: Better Vision-Language Models with Feature Adapters
    https://arxiv.org/abs/2110.04544
    """

    def build_model(self):
        # domain_names = self.data_manager.dataset.domains
        # print("Domain: ", domain_names)
    
        print("Loading CLIP Backbone: {}".format(self.cfg.MODEL.CLIPAdapters.BACKBONE))
        clip_model, _ = clip.load(
            self.cfg.MODEL.CLIPAdapters.BACKBONE,
            device=self.device,
            download_root=os.path.abspath(os.path.expanduser("data")),
        )

        print("Building Custom CLIP")
        self.model = CustomCLIP(
            self.cfg, self.data_manager.dataset.class_names, clip_model
        )

        print("Turning Off Gradients in Image and Text Encoder")
        for name, param in self.model.named_parameters():
            if "adapter" not in name:
                param.requires_grad_(False)

        # Double check
        enabled_params = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled_params.add(name)
        print("Parameters to be updated: {}".format(enabled_params))

        self.model.to(self.device)

        # NOTE: Only Give text_encoder.adapter to the Optimizer
        self.optimizer = build_optimizer(self.model.adapters, self.cfg.OPTIM)
        self.lr_scheduler = build_lr_scheduler(self.optimizer, self.cfg.OPTIM)

        # Encapsulate the three operations of optimize, simplify the process by just use the model, don't have to perform the three operations in turn
        self.model_registeration(
            "clip_adapters",
            self.model.adapters,
            self.optimizer,
            self.lr_scheduler,
        )

    def forward_backward(self, batch_data):
        # class_label.shape: torch.Size([64])
        image, class_label = self.parse_batch_train(batch_data)

        domain_label = batch_data["domain_label"]
        all_domains = self.model(image, domain_label=domain_label)
        domains_outputs = torch.split(all_domains, self.num_classes, dim=1)  # Split into 4 chunks of [batch size, 7]
        # domains_outputss = [torch.split(all_domainss[i], self.num_classes, dim=1) for i in range(len(all_domainss))]

        total_loss = 0
        losses = []

        loss_by_domain = F.cross_entropy(domains_outputs[0], class_label)
        losses.append(loss_by_domain)
        total_loss += loss_by_domain

        for dl in domain_label:
            for ith,domains_output in enumerate(domains_outputs[1:]):
                if dl==ith:
                    loss_by_domain = F.cross_entropy(domains_output, class_label)
                else:
                    loss_by_domain = -0.1 * F.cross_entropy(domains_output, class_label)
                losses.append(loss_by_domain)
                total_loss += loss_by_domain

        loss = total_loss
        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(all_domains, class_label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary