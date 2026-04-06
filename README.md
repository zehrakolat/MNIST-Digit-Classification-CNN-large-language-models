# 🧠 Handwritten Digit Recognition with CNN (MNIST)

This project is an intelligent image classification application designed to recognize and classify handwritten digits (0-9) using a **Convolutional Neural Network (CNN)**. It was developed as a final project for the **SWE015 - Introduction to Large Language Models** course.

## 🚀 Project Overview
The goal is to bridge the gap between theoretical machine learning and practical application. The system processes 28x28 grayscale images and predicts the numerical value with high precision using the **PyTorch** framework.

### Key Highlights:
* **Accuracy:** Achieved an impressive **99.15%** on the MNIST test set.
* **Architecture:** Custom CNN with two convolutional layers, max-pooling, and dropout regularization.
* **Optimization:** Utilizes the Adam optimizer with a step-learning rate scheduler and **Early Stopping** to ensure the best model weights are saved.

## 📊 Results & Performance

### 1. Classification Report
The model performs consistently across all digits, with F1-scores ranging from 0.98 to 1.00.

| Metric | Value |
| :--- | :--- |
| **Test Accuracy** | 99.15% |
| **Precision (Avg)** | 0.99 |
| **Recall (Avg)** | 0.99 |
| **F1-Score (Avg)** | 0.99 |

### 2. Visualizations
Below are the training dynamics and error analysis produced during the evaluation phase:

![Training Curve](training_curve.png) 
*Figure 1: Training and Validation loss over epochs, showing early stopping at Epoch 12.*

![Confusion Matrix](confusion_matrix.png)
*Figure 2: Confusion matrix highlighting the near-perfect classification and minor overlaps (e.g., 4 vs 9).*

## 🛠️ Tech Stack & Libraries
* **Language:** Python 3.13
* **Deep Learning:** PyTorch, Torchvision
* **Data Science:** NumPy, Scikit-learn
* **Visualization:** Matplotlib, Seaborn

## 📁 Repository Structure
* `Data_Preprocessing.py`: Handles MNIST downloading, normalization, and 60/20/20 data splitting.
* `model_arch.py`: Contains the `DigitClassifier` class (CNN architecture).
* `train.py`: The main script for training the model with Adam optimizer and Early Stopping logic.
* `evaluation.py`: Script to load the best model and generate performance metrics/graphs.
* `best_model_zehra.pth`: The saved state dictionary of the trained model.
