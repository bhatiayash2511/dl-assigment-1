# Feed Forward Neural Network Assignment

## Overview

This repository contains code for a feed forward neural network assignment. The main objective of this assignment is to build a neural network from scratch to predict classes in the Fashion MNIST dataset.

Please note that the Jupyter Notebook file `DL_Assignment1_q2_checking_mycode.ipynb` is not part of the assignment. It is included for learning purposes and does not contribute to the assignment's requirements.

## Assignment Structure

The assignment is structured as follows:

- `DL_A_Q1.ipynb`: Implementation of Question 1.
- `DL_A_Q2_Q3_Q7_implemented.ipynb`: Implementation of Questions 2, 3, and 7. This notebook is integrated with WandB.
- `train.py`: Python script with WandB integration and argparse functionality for easy parameter passing.

## Neural Network Class and Utility Functions

The core of the assignment includes:
- `neural_network` class: Implements weight initialization, backpropagation, forward propagation, and optimizer functions (SGD, NAG, Momentum, RMSProp, Adam, Nadam).
- Utility functions: Activation functions (sigmoid, relu, tanh), testing accuracy function, plotting confusion matrix of y_pred vs y and so on.

## Data Preparation

The Fashion MNIST dataset is used for this assignment. Before training the neural network, the dataset is split, and data values are normalized.

## Running the Code

To run the assignment code, follow these steps:
1. Clone this repository to your local machine.
2. Install the required dependencies, such as NumPy, Matplotlib, WandB, etc.
3. For `train.py`, use argparse to pass arguments for different configurations. Example:

python train.py -wp DL_Assignment_1 -we cs23m074 -d fashion_mnist -e 7 -b 32 -l cross_entropy -o nadam -lr 1e-3 -m 0.5 -beta 0.5 -beta1 0.5 -beta2 0.5 -eps 0.000001 -w_d 0 -w_i random -nhl 3 -sz 128 -a ReLU

# Dependencies
requirements.txt file is provided to install necessary dependencies

pip install -r requirements.txt

Execuete this in command prompt and Pip will start downloading and installing each package listed in the requirements.txt file. 

## Sweeping Configurations

In `DL_A_Q2_Q3_Q7_implemented.ipynb`, a sweep_config is provided for experimenting with different configurations. The sweep_config includes parameters for dataset, epochs, batch size, loss, optimizer, learning rate, momentum, beta, beta1, beta2, epsilon, weight decay, weight initialization, number of layers, hidden size, and activation function.

## Libraries Used

The code utilizes libraries such as NumPy, pandas, matplotlib, WandB, argparse, scikit-learn, and Keras for data manipulation, visualization, model training, and evaluation.

## Contact Information

For any questions or issues, please contact Yash Bhatia CS23M074 cs23m074@smail.iitm.ac.in +91 9039563022 

Happy coding!