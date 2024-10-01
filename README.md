# Adversarial Attack and Defense in Deep Learning

This repository contains a deep learning implementation that includes both adversarial attack techniques and a defense mechanism for a neural network model. The code utilizes PyTorch to demonstrate how adversarial examples can affect model performance and how to implement defense strategies.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Model Architecture](#model-architecture)
- [Adversarial Attacks](#adversarial-attacks)
- [Dataset Description](#dataset-description)
- [License](#license)

## Requirements

To run the code, you need the following installed:

- Python 3.6 or higher
- PyTorch
- NumPy
- Matplotlib

## Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Create a virtual environment (optional but recommended):
```
python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`
```
3. Install the required packages using pip:
```
pip install torch torchvision numpy matplotlib
```

## Usage
Run the combined script to train the model and evaluate its performance against adversarial attacks:
```
attack_defense.py

```
## Examples
### Training Example
The training process will print the loss and validation loss for each epoch. The output will look something like this:
```
Epoch: 1 Loss: 0.456 Val_Loss: 0.329
Epoch: 2 Loss: 0.298 Val_Loss: 0.278
Epoch: 3 Loss: 0.245 Val_Loss: 0.215
...
```
After training, the script will generate plots visualizing the training and validation loss over the epochs using Matplotlib.

### Attack Evaluation Example
During the evaluation, the script will output accuracy metrics based on different epsilon values, indicating how the model performs under attack. The output will look like this:
```
Epsilon: 0.0    Test Accuracy = 90 / 100 = 0.90
Epsilon: 0.007  Test Accuracy = 78 / 100 = 0.78
Epsilon: 0.01   Test Accuracy = 72 / 100 = 0.72
Epsilon: 0.02   Test Accuracy = 65 / 100 = 0.65
...
```
The script will also generate plots showing the accuracy against different epsilon values for each attack type (FGSM, IFGSM, MIFGSM).

## Model Architecture
The implemented neural network architecture consists of two convolutional networks:

```NetF``` with two convolutional layers, dropout layers, and fully connected layers.
```NetF1```, a smaller network, is used for soft label training after initial defense training.

## Adversarial Attacks
Adversarial attacks are techniques that manipulate input data to deceive machine learning models into making incorrect predictions. By introducing subtle perturbations to the input data, attackers can create adversarial examples that appear similar to legitimate inputs but lead the model to misclassify them. This highlights the vulnerabilities of deep learning models, especially in safety-critical applications like autonomous driving and facial recognition.
In this implementation, we focus on three common adversarial attack methods:
##### 1. FGSM (Fast Gradient Sign Method): An efficient one-step attack that computes the gradient of the loss function with respect to the input and adjusts the input in the direction that increases the loss.
##### 2. IFGSM (Iterative FGSM): An iterative variant of FGSM that applies the perturbation multiple times, allowing for finer control over the perturbation's strength.
##### 3. MIFGSM (Momentum Iterative FGSM): An extension of IFGSM that incorporates momentum to maintain the direction of the perturbation, enhancing the attack's effectiveness.

## Dataset Description
This implementation evaluates the models on the MNIST dataset, which consists of handwritten digits (0-9). The MNIST dataset contains 60,000 training images and 10,000 testing images, each of size 28x28 pixels in grayscale. The task is to classify each image into one of the ten digit classes.

The simplicity of the MNIST dataset makes it a popular choice for testing machine learning algorithms and provides a clear benchmark for understanding the impact of adversarial attacks on model performance.

