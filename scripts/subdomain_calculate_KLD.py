import os
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from scipy.stats import entropy
from scipy.special import kl_div
from sklearn.preprocessing import normalize
from tqdm import tqdm

torch.cuda.set_device(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(img_path):
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(img_tensor)
    return features.cpu().numpy().squeeze()

def get_domain_features(domain_path):
    features = []
    for root, _, files in os.walk(domain_path):
        for file in tqdm(files, desc=f"Processing {os.path.basename(domain_path)}"):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                features.append(extract_features(img_path))
    return np.array(features)

def kl_divergence(p, q):
    p = np.mean(p, axis=0)  # 计算平均特征向量
    q = np.mean(q, axis=0)  # 计算平均特征向量
    
    # 确保没有零值（避免log(0)的问题）
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)
    
    # 归一化
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # 使用scipy的kl_div函数计算KL散度
    kl = np.sum(kl_div(p, q))
    
    return max(0, kl)  # 确保结果非负

dataset_dir = "/data/yil708/Code-VIGIL/Week3Try/data/OfficeHome-subdomain"
domains = ["clipart0", "clipart1", "clipart2", "product0", "product1", "product2", "real_world0", "real_world1", "real_world2"]

domain_features = {}
for domain in domains:
    domain_path = os.path.join(dataset_dir, domain)
    domain_features[domain] = get_domain_features(domain_path)

kl_matrix = np.zeros((len(domains), len(domains)))
for i, domain1 in enumerate(domains):
    for j, domain2 in enumerate(domains):
        if i != j:
            kl_matrix[i, j] = kl_divergence(domain_features[domain1], domain_features[domain2])

print("KL Divergence Matrix:")
for i, domain1 in enumerate(domains):
    for j, domain2 in enumerate(domains):
        print(f"{domain1} -> {domain2}: {kl_matrix[i, j]:.4f}")

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 10))
sns.heatmap(kl_matrix, annot=True, cmap="YlGnBu", xticklabels=domains, yticklabels=domains)
plt.title("KL Divergence between Subdomains")
plt.tight_layout()
plt.savefig("kl_divergence_heatmap.png")
plt.show()