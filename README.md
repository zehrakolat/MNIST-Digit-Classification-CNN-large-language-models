Handwritten Digit Recognition with CNN (MNIST)
This project is an intelligent application designed to classify handwritten digits (0-9) using a Convolutional Neural Network (CNN). Developed as part of the SWE015 - Introduction to Large Language Models course, this project demonstrates the implementation of deep learning concepts using PyTorch.

 Features
High Accuracy: Achieved 99.15% accuracy on the MNIST test set.

Modern Architecture: Utilizes Convolutional layers, Max Pooling, and Dropout for robust feature extraction and regularization.

Optimization: Implements Adam optimizer, Learning Rate Scheduling, and Early Stopping to prevent overfitting.

Production-Ready: Modular code structure (Preprocessing, Model, Training, Evaluation).

Performance & Results
Model Metrics
The model shows exceptional performance across all classes:

Test Accuracy: 99.15%

Avg F1-Score: 0.99

Training Dynamics
The training was optimized using Early Stopping, which triggered at Epoch 12 to ensure the best generalization performance.

Tech Stack
Framework: PyTorch

Visualization: Matplotlib, Seaborn

Metrics: Scikit-learn

Dataset: MNIST (60,000 train / 10,000 test)
