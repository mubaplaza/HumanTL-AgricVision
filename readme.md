# 🌾 HumanTL-AgriVision

HumanTL-AgricVision is a deep learning–based crop classification system that leverages **Human Transfer Learning (HumanTL)** and **Convolutional Neural Networks (CNNs)** to accurately classify crop images.  
The project uses a pretrained **ResNet18** model fine-tuned for agricultural image classification and includes a **Streamlit web application** for real-time prediction.

---

## 🚀 Features
- 🌱 Crop image classification using deep learning
- 🧠 Transfer Learning with pretrained ResNet18
- 🖼️ Image preprocessing and normalization (ImageNet standard)
- 🧪 Model training, validation, and testing pipeline
- 🌐 Streamlit web app for user-friendly inference
- 💻 Optimized for **CPU systems (4GB RAM)**

---

## 🛠️ Technologies Used
- Python 3.9+
- PyTorch
- Torchvision
- Streamlit
- Pillow (PIL)
- NumPy
- Matplotlib

---

## 📂 Project Structure
HumanTL-AgriVision/
├── crop_classifier.py # Streamlit application
├── train_model.py # Training script
├── crop_classifier_model.pth # Trained PyTorch model
├── dataset/
│ ├── train/
│ ├── val/
│ └── test/
├── requirements.txt
└── README.md


## Running the Streamlit App
streamlit run crop_classifier.py

## Then open your browser at:
http://localhost:8501


## 🖼️ How It Works
1. User uploads a crop image
2. Image is resized and normalized
3. Pretrained ResNet18 extracts features
4. Fine-tuned classifier predicts crop type
5. Result is displayed instantly

## 📊 Model Details

- Architecture: ResNet18
- Learning Method: Transfer Learning (HumanTL)
- Loss Function: CrossEntropyLoss
- Optimizer: Adam
- Input Size: 224 × 224 RGB images

## Notes
- Designed to run efficiently on CPU-only systems
- Batch size optimized for low-memory environments
- Model trained using ImageNet normalization