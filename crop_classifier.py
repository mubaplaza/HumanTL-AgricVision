import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torch.nn as nn

# -----------------------------
# Device (CPU only)
# -----------------------------
device = torch.device("cpu")

# -----------------------------
# Load model
# -----------------------------
checkpoint = torch.load(
    "crop_classifier_model.pth",
    map_location=device
)

num_classes = len(checkpoint["class_to_idx"])

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

idx_to_class = {v: k for k, v in checkpoint["class_to_idx"].items()}

# -----------------------------
# Image transforms
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🌾 Crop Classification🌾")
st.markdown("Upload a crop image and the model will predict the crop type.")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=400)


    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        class_name = idx_to_class[predicted.item()]

    st.success(f"Predicted Crop: **{class_name}**")
