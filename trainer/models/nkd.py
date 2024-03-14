import os

import timm
import torch
from clip import clip
from torch.nn import functional as F

from metrics import compute_accuracy
from optim import build_lr_scheduler, build_optimizer
from trainer import MODEL_REGISTRY, Trainer
from utils import PROMPT_TEMPLATES


@MODEL_REGISTRY.register()
class NKD(Trainer):
    """NKD

    A Naive Knowledge Distillation Approach.
    """

    def build_model(self):
        print("Loading CLIP Backbone: {}".format(self.cfg.MODEL.NKD.BACKBONE))
        self.clip_model, _ = clip.load(
            self.cfg.MODEL.NKD.BACKBONE,
            device=self.device,
            download_root=os.path.abspath(os.path.expanduser("data")),
        )

        self.text_feature_dim = 512

        # Construct Prompts
        prompt_template = PROMPT_TEMPLATES[self.cfg.DATASET.NAME]
        prompts = [
            prompt_template.format(class_name.replace("_", " "))
            for class_name in self.data_manager.dataset.class_names
        ]
        prompts = torch.cat([clip.tokenize(prompt) for prompt in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            self.clip_model.eval()
            self.text_features = self.clip_model.encode_text(prompts)
            self.text_features_norm = self.text_features / self.text_features.norm(
                dim=-1, keepdim=True
            )

        self.student_model = timm.create_model(
            self.cfg.MODEL.NKD.STUDENT_NETWORK,
            pretrained=True,
            num_classes=self.num_classes,
        ).to(self.device)

        self.optimizer = build_optimizer(self.student_model, self.cfg.OPTIM)
        self.lr_scheduler = build_lr_scheduler(self.optimizer, self.cfg.OPTIM)
        self.model_registeration(
            "nkd",
            self.student_model,
            self.optimizer,
            self.lr_scheduler,
        )

        self.distillation_loss_weight = self.cfg.MODEL.NKD.LOSS_WEIGHT.DISTILLATION
        self.classification_loss_weight = self.cfg.MODEL.NKD.LOSS_WEIGHT.CLASSIFICATION
        self.temperature = self.cfg.MODEL.NKD.TEMPERATURE

        print("Distillation_Loss_Weight: {}".format(self.distillation_loss_weight))
        print("Classification_Loss_Weight: {}".format(self.classification_loss_weight))
        print("Temperature: {}".format(self.temperature))

    def forward_backward(self, batch_data):
        image, class_label = self.parse_batch_train(batch_data)

        # Compute Image Features for both Teacher (CLIP) and Student
        with torch.no_grad():
            self.clip_model.eval()
            teacher_image_features = self.clip_model.encode_image(image)

        teacher_image_features = teacher_image_features / teacher_image_features.norm(
            dim=-1, keepdim=True
        )
        logit_scale = self.clip_model.logit_scale.exp()
        teacher_logits = (
            logit_scale * teacher_image_features @ self.text_features_norm.T
        )
        student_image_features = self.student_model.forward_features(image)
        student_image_features = self.student_model.global_pool(student_image_features)
        student_logits = self.student_model.fc(student_image_features)

        # --- Classification Loss
        classification_loss = F.cross_entropy(student_logits, class_label)

        # --- Distillation Loss
        distillation_loss = (
            F.kl_div(
                F.log_softmax(student_logits / self.temperature, dim=1),
                F.softmax(teacher_logits / self.temperature, dim=1),
                reduction="batchmean",
            )
            * self.temperature
            * self.temperature
        )

        loss = (
            distillation_loss * self.distillation_loss_weight
            + classification_loss * self.classification_loss_weight
        )

        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "distil_loss": distillation_loss.item(),
            "class_loss": classification_loss.item(),
            "acc": compute_accuracy(student_logits, class_label)[0].item(),
        }

        return loss_summary

    def model_inference(self, input_data):
        return self.student_model(input_data)
