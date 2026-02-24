# Aerial Object Classification (Bird vs Drone)

This project is a deep learning application that classifies aerial images
as either **Bird** or **Drone** using a convolutional neural network.

The model is built using **ResNet50 with transfer learning** and deployed
as a **Streamlit web application**.

---

## 📌 Problem Statement
Distinguishing birds from drones in aerial images is challenging because
both can have similar shapes, sizes, and backgrounds. This project aims
to solve that problem using deep learning-based image classification.

---

## 🧠 Model & Approach
- Model: ResNet50 (CNN with residual connections)
- Transfer learning using ImageNet weights
- Fine-tuning of deeper layers for better feature learning
- Data augmentation to improve generalization
- Class imbalance handled using class weights

---

## 🗂 Dataset Structure
The dataset is organized as follows:


train/
├── bird/
└── drone/

valid/
├── bird/
└── drone/

test/
├── bird/
└── drone/


---

## 🚀 How to Run the Project

### 1️⃣ Install required libraries
```bash
pip install -r requirements.txt
2️⃣ Run the Streamlit application
streamlit run app.py
📊 Application Output

Predicts whether the uploaded image is Bird or Drone

Displays prediction confidence

Handles low-confidence predictions gracefully

📁 Project Files

app.py – Streamlit application

models/ – Saved trained model (.keras)

notebook/ – Model training notebook

requirements.txt – Python dependencies

🧪 Optional Analysis

Confusion Matrix

Classification Report

Accuracy and Loss plots

Grad-CAM visualization for model explainability

✅ Conclusion

This project demonstrates how deep learning and transfer learning can be
used to solve a real-world aerial image classification problem and deploy
it as an interactive web application.