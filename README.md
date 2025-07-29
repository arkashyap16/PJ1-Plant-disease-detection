# 🌿 Plant Disease Detection using Deep Learning

This project aims to accurately detect plant diseases using deep learning and computer vision. It leverages a CNN model trained on image datasets of diseased and healthy plant leaves, and provides predictions along with confidence scores via an interactive Streamlit web application.

## 📌 Features

- 🌱 Classifies plant leaf images into 5 disease categories
- 🧠 Trained with Convolutional Neural Networks (CNN)
- 📊 Model evaluation with ~86.65% test accuracy
- 🖼️ Real-time prediction using uploaded leaf images
- 🎯 Displays prediction confidence
- 🌐 Interactive UI using Streamlit

---

## 🧠 Model Details

- Framework: TensorFlow / Keras
- Architecture: CNN with convolution, pooling, and dense layers
- Loss Function: `categorical_crossentropy`
- Optimizer: `Adam`
- Metrics: Accuracy

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/plant-disease-detector.git
cd plant-disease-detector

- Run the web app
streamlit run CODE/app.py

