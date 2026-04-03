# mnist-logistic-softmax-from-scratch

## Overview

This project implements **Logistic Regression** and **Softmax Regression** for image classification on the MNIST dataset.

* Binary classification (digits **0 vs 1**) using Logistic Regression *(from scratch with NumPy)*
* Multi-class classification (**0–9**) using Softmax Regression *(from scratch)*
* Comparison with Scikit-learn implementations

---

## Dataset

* MNIST dataset
* 60,000 training images
* 10,000 testing images
* Image size: **28×28 pixels**

---

## Implementations

### 1. Logistic Regression (NumPy)

* Implemented from scratch
* Optimized using Gradient Descent

**Evaluation metrics:**

* Precision
* Recall
* F1-score

---

### 2. Softmax Regression (NumPy)

* Multi-class classification
* One-hot encoding
* Cross-entropy loss
* Gradient Descent optimization

---

### 3. Scikit-learn Models

* Logistic Regression (binary classification)
* Multinomial Logistic Regression (Softmax)

---

## Results

* Precision, Recall, F1-score
* Loss visualization during training

---

## How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run notebook

```bash
jupyter notebook
```

---

## Project Structure

```
mnist-logistic-softmax-from-scratch/
│
├── notebooks/
│   └── mnist_logistic_softmax.ipynb
|
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Key Learnings

* Understanding Gradient Descent
* Binary vs Multi-class classification
* Implementing Softmax from scratch
* Comparing NumPy vs Scikit-learn

---

## Author
* Nguyễn Minh Triết
* 24521853

* Your Name
* Student ID: Your ID
