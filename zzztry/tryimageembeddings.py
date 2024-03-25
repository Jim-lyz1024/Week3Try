import torch
from PIL import Image
from clip import clip
import torchvision.transforms as transforms

# Load the CLIP model
clip_model, _ = clip.load("ViT-B/32", device="cpu")

# Define a transform to preprocess the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

# Load and preprocess an image
image_path = "./74a456594b63469e8745aa6cfd68ca40.jpeg"  # Replace with your image path
image = transform(Image.open(image_path)).unsqueeze(0)  # Add batch dimension

# Process the image through the CLIP image encoder
with torch.no_grad():
    image_features = clip_model.encode_image(image)

# Print the image features
print("Image features:", image_features)
