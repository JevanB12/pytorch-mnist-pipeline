# 🧠 AI Pipeline Tutorial with PyTorch (MNIST)

## 📌 Abstract
This tutorial demonstrates how to build and train a machine learning pipeline using PyTorch within a Jupyter Notebook environment. It walks through dataset preparation, neural network design, training, and evaluation using the MNIST dataset. Additionally, it explores how hyperparameters affect performance and visualises results using graphs and tables. By the end, readers will understand how to implement and analyse a complete AI pipeline suitable for real-world applications.

---

## 🎯 Learning Objectives
By following this tutorial, you will:
- Understand the structure of an AI pipeline in Jupyter Notebook
- Learn how to use PyTorch for building neural networks
- Train a model using the MNIST dataset
- Evaluate performance using metrics, graphs, and tables
- Analyse how hyperparameters affect model accuracy
- Compare approaches with existing tutorials

---

## 📚 Table of Contents
1. Introduction  
2. Related Work (Comparison with Tutorials)  
3. Setup & Dependencies  
4. Dataset (MNIST)  
5. Model Architecture  
6. Training Process  
7. Hyperparameter Tuning  
8. Results & Visualisation  
9. Discussion  
10. Conclusion  
11. References  

---

## ⚙️ Technologies Used
- Python  
- PyTorch  
- NumPy  
- Matplotlib  
- Jupyter Notebook  

---

## 📊 Dataset
We use the **MNIST dataset**, a standard benchmark for handwritten digit classification.

---

## 🧩 Pipeline Overview
The AI pipeline includes:
- Data loading and preprocessing  
- Model definition (Neural Network)  
- Training loop with optimisation  
- Evaluation on test data  
- Visualisation of results  

---

## 🏗️ Model Architecture
```python
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
