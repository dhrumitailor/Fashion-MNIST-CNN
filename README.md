# 👕 Fashion-MNIST Clothes Classification using CNN

A deep learning project that classifies clothing images from the Fashion-MNIST dataset into 10 categories using a Convolutional Neural Network (CNN) built with TensorFlow and Keras.

---

## 📌 Project Overview

The objective of this project is to classify grayscale clothing images into one of ten fashion categories using Computer Vision and Deep Learning.

The model is trained on the **Fashion-MNIST** dataset and achieves approximately **90% test accuracy**.

---

## 🎯 Objectives

- Load and preprocess the Fashion-MNIST dataset.
- Build a Convolutional Neural Network (CNN).
- Train the model using TensorFlow/Keras.
- Evaluate model performance.
- Generate a Confusion Matrix and Classification Report.
- Develop a Streamlit web application for interactive predictions.

---

## 🧠 Dataset

Dataset: **Fashion-MNIST**

- 70,000 grayscale images
- Image size: **28 × 28**
- 10 clothing categories
- 60,000 training images
- 10,000 testing images

Dataset Source:
https://github.com/zalandoresearch/fashion-mnist

---

## 👕 Classes

| Label | Category |
|-------|----------|
| 0 | T-shirt/Top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle Boot |

---

## 🛠 Technologies Used

- Python 3.11
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit

---

## 🏗 CNN Architecture

Input Layer
↓
Conv2D (32 Filters, 3×3, ReLU)
↓
Batch Normalization
↓
MaxPooling2D
↓
Dropout (0.25)
↓
Conv2D (64 Filters, 3×3, ReLU)
↓
Batch Normalization
↓
MaxPooling2D
↓
Dropout (0.25)
↓
Flatten
↓
Dense (128, ReLU)
↓
Dropout (0.5)
↓
Dense (10, Softmax)

---

## 📊 Model Performance

**Test Accuracy**

```
90.38%
```

**Test Loss**

```
0.2745
```

### Best Performing Classes

- Trouser
- Bag
- Sneaker
- Sandal
- Ankle Boot

### Most Challenging Class

- Shirt

The confusion matrix shows that the model mainly confuses:

- Shirt
- T-shirt/Top
- Pullover
- Coat

because these classes have very similar visual appearances in low-resolution grayscale images.

---

## 📈 Results

The project includes:

- Training Accuracy Graph
- Validation Accuracy Graph
- Training Loss Graph
- Validation Loss Graph
- Confusion Matrix
- Classification Report

---

## 🌐 Streamlit Web Application

A Streamlit application is included to demonstrate the trained CNN.

Features:

- Browse Fashion-MNIST test images
- Predict clothing category
- Display confidence score
- Show actual and predicted labels
- Display Top-5 prediction probabilities

Run the application using:

```bash
streamlit run app.py
```

---

## 🚀 How to Run

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/Fashion-MNIST-CNN.git
```

### Move into Project

```bash
cd Fashion-MNIST-CNN
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train the Model

```bash
python train.py
```

### Launch Streamlit App

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
Fashion-MNIST-CNN/
│
├── train.py
├── app.py
├── predict.py
├── cnn_model.keras
├── requirements.txt
├── README.md
└── screenshots/
```

---

## 📚 Learning Outcomes

This project demonstrates:

- Image preprocessing
- Convolutional Neural Networks (CNNs)
- Feature extraction using Conv2D
- Batch Normalization
- Max Pooling
- Dropout Regularization
- Model Evaluation
- Confusion Matrix Analysis
- Classification Report
- Model Deployment using Streamlit

---

## 👨‍💻 Author

**Anshu**

B.Tech Information Technology

---

## ⭐ Future Improvements

- Train on real-world clothing image datasets.
- Improve accuracy using data augmentation.
- Apply transfer learning (MobileNetV2 / EfficientNet).
- Deploy the application on Streamlit Cloud.

---

## 📄 License

This project is developed for educational and learning purposes.