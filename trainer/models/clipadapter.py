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
        self.adapter = Adapter(512, 4).to(clip_model.dtype)
        self.dtype = clip_model.dtype
        self.mode = 'train'  # default mode
        self.cfg = cfg
        self.class_names = class_names
        self.clip_model = clip_model
        self.text_features = {}
        self.update_text_features(self.cfg)
        
    def update_text_features(self, cfg):
        domain_names = cfg.DATASET.SOURCE_DOMAINS if self.mode == 'train' else cfg.DATASET.TARGET_DOMAINS
        prompt_template = PROMPT_TEMPLATES[cfg.DATASET.NAME]
        
        # Generate prompts for each domain
        prompts_domain = {}
        
        prompts_original = [prompt_template.format(class_name.replace("_", " ")) for class_name in self.class_names]
        prompts_domain['original'] = prompts_original
        
        for domain in domain_names:
            prompts_domain[domain] = [
            prompt_template.format(domain.replace("_", " ") + ' ' + class_name.replace("_", " "))
            for class_name in self.class_names
        ]
        
        print(prompts_domain)
        # exit()
        
        self.text_features = {}
        for domain, prompts in prompts_domain.items():
            tokenized_prompts = [clip.tokenize(prompt) for prompt in prompts]
            # Flatten the list of tokenized prompts
            tokenized_prompts = torch.cat(tokenized_prompts).to(torch.cuda.current_device())
            
            # Obtain text features for each domain's prompts
            with torch.no_grad():
                self.text_features[domain] = self.clip_model.encode_text(tokenized_prompts)
                self.text_features[domain] = self.text_features[domain] / self.text_features[domain].norm(dim=-1, keepdim=True)

    def forward(self, image):
        if self.mode == 'eval' and not hasattr(self, 'eval_mode_set'):
            self.update_text_features(self.cfg)
            self.eval_mode_set = True  # Set a flag to avoid re-updating text features unnecessarily
            print(self.text_features)

        adapter_ratio = 0.2
        # computes the image features using the CLIP image encoder
        image_features = self.image_encoder(image.type(self.dtype))
        # obtain adapted features
        adapter_features = self.adapter(image_features)
    
        image_features = (
            adapter_ratio * adapter_features + (1 - adapter_ratio) * image_features
        )
        # regularization, avoid updating gradient too fast
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Image features shape: torch.Size([64, 512])
        
        # Calculate similarity
        logit_scale = self.logit_scale.exp()
        
        # Calculate logits for each domain
        # logits = logit_scale * image_features @ self.text_features.t() # .t() means transpose of a matrix
        logits_domain = {}
        for domain, text_feature in self.text_features.items():
            logits_domain[domain] = logit_scale * image_features @ text_feature.t()
        
        all_domains = torch.cat(list(logits_domain.values()), dim=1) # All Domains: torch.Size([64, 28])

        return all_domains # logits_domain 
    
 
@MODEL_REGISTRY.register()
class CLIPAdapter(Trainer):
    """CLIP-Adapter

    CLIP-Adapter: Better Vision-Language Models with Feature Adapters
    https://arxiv.org/abs/2110.04544
    """

    def build_model(self):
        # domain_names = self.data_manager.dataset.domains
        # print("Domain: ", domain_names)
    
        print("Loading CLIP Backbone: {}".format(self.cfg.MODEL.CLIPAdapter.BACKBONE))
        clip_model, _ = clip.load(
            self.cfg.MODEL.CLIPAdapter.BACKBONE,
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
        self.optimizer = build_optimizer(self.model.adapter, self.cfg.OPTIM)
        self.lr_scheduler = build_lr_scheduler(self.optimizer, self.cfg.OPTIM)

        # Encapsulate the three operations of optimize, simplify the process by just use the model, don't have to perform the three operations in turn
        self.model_registeration(
            "clip_adapter",
            self.model.adapter,
            self.optimizer,
            self.lr_scheduler,
        )

    def forward_backward(self, batch_data):
        image, class_label = self.parse_batch_train(batch_data)
        all_domains = self.model(image)

        domains_outputs = torch.split(all_domains, self.num_classes, dim=1)  # Split into 4 chunks of [batch size, 7]
        # print("Domains Outputs:", domains_outputs)

        total_loss = 0
        losses = []

        # Compute the loss for each domain
        for domain_output in domains_outputs:
            loss_by_domain = F.cross_entropy(domain_output, class_label)
            losses.append(loss_by_domain)
            total_loss += loss_by_domain 
        
        loss = total_loss / len(domains_outputs)                                                      
            
        # output = self.model(image)
        # loss = F.cross_entropy(output, class_label)
        
        """ logits_domain = self.model(image)
        
        print("Logits domain:", logits_domain)
        
        # Initialize a dictionary to store loss for each domain
        losses_domain = {}
        total_loss = 0
        
        for domain_name, output in logits_domain.items():
            # print(f"Domain: {domain_name}, Output: {output}")
            loss_by_domain = F.cross_entropy(output, class_label)
            losses_domain[domain_name] = loss_by_domain
            total_loss += loss_by_domain
        loss = total_loss / len(losses_domain) """

        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(all_domains, class_label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary
