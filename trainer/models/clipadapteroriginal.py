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
        
        domain_names = cfg.DATASET.SOURCE_DOMAINS
        print("Class:", class_names)
        print("Domains:", domain_names)

        # Construct Prompts
        prompt_template = PROMPT_TEMPLATES[cfg.DATASET.NAME]
        
        prompts = [
            prompt_template.format(class_name.replace("_", " "))
            for class_name in class_names
        ]
        
        # prompts = [
        #     prompt_template.format(domain_name.replace("_", " ") + ' ' + class_name.replace("_", " "))
        #     for domain_name in domain_names for class_name in class_names
        # ]
        
        print(prompts)
        print("Number of prompts:", len(prompts))
        # exit()
        
        prompts = torch.cat([clip.tokenize(prompt) for prompt in prompts])
        prompts = prompts.to(torch.cuda.current_device())

        with torch.no_grad():
            self.text_features = clip_model.encode_text(prompts)
            self.text_features = self.text_features / self.text_features.norm(
                dim=-1, keepdim=True
            )
        
        # print("Text features:", self.text_features)
        print("Text features shape:", self.text_features.shape) # Text features shape: torch.Size([28, 512])
        exit()

    def forward(self, image):
        adapter_ratio = 0.2
        # computes the image features using the CLIP image encoder
        image_features = self.image_encoder(image.type(self.dtype))
        # obtain adapted features
        adapter_features = self.adapter(image_features)
        
        # print("Image features:", image_features)
        # print("Adapter features:", adapter_features)
    
        image_features = (
            adapter_ratio * adapter_features + (1 - adapter_ratio) * image_features
        )
        # regularization, avoid updating gradient too fast
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Image features shape: torch.Size([64, 512])
        # print("Image features shape:", image_features.shape) 

        # Calculate similarity
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t() # .t() means transpose of a matrix
        
        # print("Logits: ", logits.shape) # Logits:  torch.Size([64, 28])
        topk_values, topk_indices = torch.topk(logits, k=4, dim=1)

        # print("Top 4 scores for each image:", topk_values)
        # print("Top 4 class indices for each image:", topk_indices)

        return logits


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
        output = self.model(image)
        loss = F.cross_entropy(output, class_label)

        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, class_label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary
