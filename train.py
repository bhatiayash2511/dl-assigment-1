# Importing libraries
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import wandb
import argparse
from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist
from keras.datasets import mnist
from sklearn.metrics import confusion_matrix



parser = argparse.ArgumentParser()
parser.add_argument("-wp","--wandb_project",help="Project name used to track experiments in Weights & Biases dashboard",default="DL_Assigment_1")
parser.add_argument("-we","--wandb_entity",help="Wandb Entity used to track experiments in the Weights & Biases dashboard.",default="cs23m074")
parser.add_argument("-d","--dataset",help=f"choices: ['fashion_mnist', 'mnist']",choices=['fashion_mnist','mnist'],default='fashion_mnist')
parser.add_argument("-e","--epochs",help="Number of epochs to train neural network.",choices=['5','10'],default=7)
parser.add_argument("-b","--batch_size",help="Batch size used to train neural network.",choices=['16','32','64'],default=32)
parser.add_argument("-l","--loss",help=f"choices: ['cross_entropy', 'mean_squared_error']",choices=['cross_entropy', 'mean_squared_error'],default='cross_entropy')
parser.add_argument("-o","--optimizer",help=f"choices: ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']",choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'],default='nadam')
parser.add_argument("-lr","--learning_rate",help="Learning rate used to optimize model parameters",choices=['1e-3','1e-4'],default=1e-3)
parser.add_argument("-m","--momentum",help="Momentum used by momentum and nag optimizers.",choices=['0.5'],default=0.5)
parser.add_argument("-beta","--beta",help="Beta used by rmsprop optimizer",choices=['0.5'],default=0.5)
parser.add_argument("-beta1","--beta1",help="Beta1 used by adam and nadam optimizers.",choices=['0.5'],default=0.5)
parser.add_argument("-beta2","--beta2",help="Beta2 used by adam and nadam optimizers.",choices=['0.5'],default=0.5)
parser.add_argument("-eps","--epsilon",help="Epsilon used by optimizers.",choices=['0.000001'],default=0.000001)
parser.add_argument("-w_d","--weight_decay",help="Weight decay used by optimizers.",choices=['0','0.0005','0.5'],default=0)
parser.add_argument("-w_i","--weight_init",help=f"choices: ['random', 'Xavier']",choices=['random', 'Xavier'],default='random')
parser.add_argument("-nhl","--num_layers",help="Number of hidden layers used in feedforward neural network.",choices=['3','4','5'],default=3)
parser.add_argument("-sz","--hidden_size",help="Number of hidden neurons in a feedforward layer.",choices=['32','64','128'],default=128)
parser.add_argument("-a","--activation",help=f"choices: ['sigmoid', 'tanh', 'ReLU']",choices=['sigmoid', 'tanh', 'ReLU'],default='ReLU')
args = parser.parse_args()




# Loading Dataset according to argparser and splitting

print("Loading Data...")
if args.dataset == "fashion_mnist":
  (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
else:
  (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
print(" Dataset Loaded !!")

# Normalizing values and reshaping training values

def normalizeX(x_train, x_test, x_val):
  x_train = x_train / 255
  x_test = x_test / 255
  x_val = x_val / 255
  return x_train, x_test, x_val

x_train, x_test, x_val = normalizeX(x_train, x_test, x_val)

def shapeX(x_train, x_test, x_val):
  x_train = x_train.reshape(x_train.shape[0], -1)
  x_test = x_test.reshape(x_test.shape[0], -1)
  x_val = x_val.reshape(x_val.shape[0], -1)
  return x_train, x_test, x_val

x_train, x_test, x_val = shapeX(x_train, x_test, x_val)
input_size = x_test.shape[1]


def sigmoid( x):
    return np.tanh(x)
    temp2 = np.exp(-x)
    temp = 1.0/(1.0 + temp2)
    return temp

def sigmoid_derivative( x):
    return 1-np.tanh(x)**2
    sigm = sigmoid(x)
    temp = sigm
    temp2 = 1 - sigm 
    return sigm*(1-sigm)


def relu( x):
    return np.where(x > 0, x, 0)


def relu_derivative( x):
    return np.where(x < 0, 0, 1)



def tanh( x):
    t = np.tanh(x)
    return t 


def tanh_derivative( x):
    temp = np.tanh(x)
    temp2 = 1 - temp*temp
    return temp2


  

def softmax( x):
    i = 0
    while i < x.shape[0]:
        argmax = np.argmax(x[i])
        sum = 0
        maxval = x[i][argmax]
        j = 0
        while j < x.shape[1]:
            sum = sum + np.exp(x[i][j]-maxval)
            j += 1
        x[i] = np.exp(x[i]-maxval)/sum
        i+=1
    return x
  
def softmax_derivative( x):
    temp = softmax(x)
    temp2 = 1 - temp
    return temp*temp2


def one_hot_encoded( y, size):
    temp = np.eye(size)[y]
    return temp


def cross_entropy( y_train, y_hat):
    loss = 0
    i = 0
    while i < y_hat.shape[0]:
        loss += -(np.log2(y_hat[i][y_train[i]]))
        i += 1
    return loss/y_hat.shape[0]

def batch_converter(x1, y1, batch_size1):
    x, y, batch_size = x1, y1, batch_size1
    x_batch = []
    y_batch = []
    num_datapoints = x.shape[0]
    no_datapoints = num_datapoints
    no_batches = no_datapoints // batch_size                   # floor division operator
    i = 0
    while i < no_batches:
        e = 0
        if (i+1)*batch_size < x.shape[0]:
            e = (i+1)*batch_size
        else:
            e = x.shape[0]
        s = i*batch_size
        x1 = np.array(x[s:e])        # slicing
        y1 = np.array(y[s:e])        # slicing
        x_batch.append(x1)
        y_batch.append(y1)
        i += 1
    
    temp = no_batches * batch_size
    if temp != x_train.shape[0]:
        x1 = np.array(x_train[temp :])
        y1 = np.array(y_train[temp :])
        x_batch.append(x1)
        y_batch.append(y1)
    return x_batch, y_batch


def squared_error( y_train, y_hat, no_of_classes):
    y_onehot = one_hot_encoded(y_train, no_of_classes)
    loss = 0
    i = 0
    while i < y_hat.shape[0]:
        loss += np.sum((y_hat[i] - y_onehot[i])**2)
        i+=1
    return loss / y_train.shape[0]

class neural_network:
  def __init__(self, s, weight_initialisation):
    self.W, self.B, self.preactivation, self.activation = [],[],[],[]
    self.initializer = weight_initialisation
    self.network = s
    self.initializeWandB()

  def initializeWandB(self):
    if self.initializer.lower() != "random":
      i = 1
      lengtht = len(self.network)
      while i < lengtht:
        temp = 6 / (self.network[i] + self.network[i-1])
        n = np.sqrt(temp)
        w = np.random.uniform(-n , n, (self.network[i], self.network[i-1]))
        self.W.append(w)
        b = np.random.uniform(-n , n, (self.network[i]))
        self.B.append(b)
        i += 1
    # Random weight Initialise
    elif self.initializer.upper() != "XAVIER":
      i = 1
      while i < len(self.network):
        b = np.random.randn(self.network[i])
        self.B.append(b)
        w = np.random.randn(self.network[i], self.network[i-1]) /(np.sqrt(self.network[i]))
        self.W.append(w)
        i += 1

  

  def loss_function(self, y_train, y_hat, no_of_classes, loss_func, lambd):
    temp = y_train.shape[0]
    loss = self.l2_regularize(lambd, temp)
    if loss_func.upper() != "CROSS_ENTROPY":
      loss = loss + squared_error(y_train, y_hat, no_of_classes)
    else:
      loss = loss + cross_entropy(y_train, y_hat)
    return loss



  


  def l2_regularize(self, lambd, batch_size):
    acc = 0
    i = 0
    while i < len(self.W):
      acc += np.sum(self.W[i] ** 2)
      i += 1
    temp = (lambd/(2.* batch_size)) * acc
    return temp

  def forward(self, input, size1, activation_function1):
    # Hiddlen layers
    size, activation_function = size1, activation_function1
    i = 0
    temp2 = activation_function.upper()
    temp = len(size)-1
    while i < temp:
      Y = np.dot(input, self.W[i].T) + self.B[i]
      
      '''Not normalizing'''
      if i < len(size)-2:
        if i < len(self.preactivation):
          self.preactivation[i] = Y
        else:
          self.preactivation.append(Y)
        
        if temp2 == "RELU":
          Z = relu(Y)
        if temp2 =="SIGMOID":
          Z = sigmoid(Y)
        if temp2 =="TANH":
          Z = tanh(Y)
        
        temp3 = len(self.activation)
        if i < temp3:
          self.activation[i] = Z
        else:
          self.activation.append(Z)
        input = Z
      else:
        # Output layer.
        Y = np.dot(input, self.W[i].T) + self.B[i]
        
        temp4 =len(self.preactivation)
        if i < temp4:
          self.preactivation[i] = Y
        else:
          self.preactivation.append(Y)
        Z = softmax(Y)
        temp5 = len(self.activation)
        if i < temp5:
          self.activation[i] = Z
        else:
          self.activation.append(Z)
      i = i + 1
    return self.preactivation, self.activation


  def backward(self, layers1, x1, y1 ,no_of_classes1, preac1, ac1, activation_function1, loss_func1):
    layers, x, y ,no_of_classes, preac, ac, activation_function, loss_func = layers1, x1, y1 ,no_of_classes1, preac1, ac1, activation_function1, loss_func1
    no_layers = len(layers)
    grad_a, grad_w, grad_b, grad_h = [],[],[],[]
    y_onehot = one_hot_encoded(y, no_of_classes)
    temp4 = activation_function.upper()
    temp = "cross_entropy"
    temp2 = len(ac)-1
    if loss_func.lower() == temp:

      temp = -(y_onehot - ac[temp2])
      grad_a.append(temp)
    else:
      grad_a.append((ac[temp2] - y_onehot) * softmax_derivative(ac[temp2]))
    i = no_layers - 2
    while i  > -1:
      temp3 = no_layers-2-i
      if i == 0:
        dw = (grad_a[temp3].T @ x)
        db = np.sum(grad_a[temp3],axis=0)/y.shape[0]
      elif i > 0:
        dw = (grad_a[temp3].T @ ac[i-1])
        db = np.sum(grad_a[temp3],axis=0)/ y.shape[0]
        dh_1 = grad_a[temp3] @ self.W[i]
        sig = 0
        if temp4 == "SIGMOID":
          sig = sigmoid_derivative(preac[i-1])
        if temp4 == "RELU":
          sig = relu_derivative(preac[i-1])
        if temp4 == "TANH":
          sig = tanh_derivative(preac[i-1])
        

        da_1 = dh_1 * sig

        grad_h.append(dh_1)
        grad_a.append(da_1)
      grad_b.append(db)
      grad_w.append(dw)
      i -= 1
    return grad_w, grad_b


  def printingFunction(self, loss_train, loss_val, accur_train, accur_val, i, optimizer_name):
      print()
      print(i+1, "Iteration No : ", "\t Train Loss\t", loss_train)
      print(i+1, "Iteration No : ", "\t Validate Loss\t", loss_val)
      print(i+1, "Iteration No : ", "\t Train Accuracy\t", accur_train)
      print(i+1, "Iteration No : ", "\t Validate Accuracy\t", accur_val)
      print("---------------------------------------------------------")

  def batch_grad_descent(self, x_train, y_train, x_test, y_test, no_of_classes, layers, activation_function, eta, batch_size, n_iterations, loss_func, lambd, do_wandb_log):
    x_batch, y_batch = batch_converter(x_train, y_train, batch_size)
    loss_arr = []
    i = 0
    length = len(layers)
    while i < n_iterations:
      j = 0
      while j < len(x_batch):
        xb, yb = x_batch[j], y_batch[j]
        preac, ac = None, None
        def f(xb, layers, activation_function, yb, no_of_classes, preac, ac, loss_func):
          preac, ac = self.forward(xb, layers, activation_function)
          grad_w, grad_b = self.backward(layers, xb, yb, no_of_classes,preac, ac, activation_function, loss_func)
          return preac, ac, grad_w, grad_b
        preac, ac, grad_w, grad_b = f(xb, layers, activation_function, yb, no_of_classes, preac, ac, loss_func)
        l = 0
        while l < length-1:
          self.W[l] += -(eta * grad_w[length-l-2] + eta * lambd * self.W[l])
          self.B[l] += -eta * grad_b[length-l-2]
          l += 1
        j += 1

      loss_train = 0
      loss_val = 0

      def f2(x_train, layers, activation_function, y_train, ac, no_of_classes, loss_func, lambd, x_test, y_test):
        preac, ac = self.forward(x_train, layers, activation_function)
        loss_train = self.loss_function(y_train, ac[len(ac)-1], no_of_classes, loss_func, lambd)

        preac, ac = self.forward(x_test, layers, activation_function )
        loss_val = self.loss_function(y_test, ac[len(ac)-1], no_of_classes, loss_func, lambd)
        return loss_train, loss_val

      loss_train, loss_val = f2(x_train, layers, activation_function, y_train, ac, no_of_classes, loss_func, lambd, x_test, y_test)
      loss_arr.append(loss_val)
      accur_train = self.test_accuracy(layers, x_train, y_train, activation_function)
      accur_val = self.test_accuracy(layers, x_test, y_test, activation_function)

      name = "SGD"
      self.printingFunction(loss_train, loss_val, accur_train, accur_val, i, name)


      
      if do_wandb_log == True:
        wandb_plots = dict({"epoch": i+1 , "train_accuracy":accur_train,"train_error":loss_train,"val_accuracy":accur_val,"val_error":loss_val})
        wandb.log(wandb_plots)
      i += 1
    return ac[len(ac)-1], y_test



  def momentum_grad_descent(self, x_train, y_train, x_test, y_test, no_of_classes, layers, activation_function, batch_size, eta, epochs, beta, loss_func,lambd, do_wandb_log):
    l = len(layers)
    prev_w, prev_b, loss_arr = [],[],[]
    i = 0
    while i < l-1:
      prev_w.append(np.zeros(self.W[i].shape))
      prev_b.append(np.zeros(self.B[i].shape))
      i += 1
    x_batch, y_batch = batch_converter(x_train, y_train, batch_size)
    ep = 0
    while ep < epochs:
      j = 0
      while j < len(x_batch):
        xb = x_batch[j]
        yb = y_batch[j]
        preac, ac = self.forward(xb, layers, activation_function)
        grad_w, grad_b = self.backward(layers, xb, yb, no_of_classes, preac, ac, activation_function, loss_func)
        i = 0
        while i < l-1:
          prev_w[i] = beta*prev_w[i] + grad_w[l-i-2]
          prev_b[i] = beta*prev_b[i] + grad_b[l-i-2]
          self.W[i] += -(eta*prev_w[i] + eta * lambd * self.W[i])
          self.B[i] += -eta*prev_b[i]
          i += 1
        j += 1

      loss_train = 0
      loss_val = 0

      def f3(x_train, layers, activation_function, y_train, ac, no_of_classes, loss_func, lambd, x_test, y_test):

        preac, ac = self.forward(x_train, layers, activation_function)
        loss_train = self.loss_function(y_train, ac[len(ac)-1], no_of_classes, loss_func, lambd)

        preac, ac = self.forward(x_test, layers, activation_function)
        loss_val = self.loss_function(y_test, ac[len(ac)-1], no_of_classes, loss_func, lambd)
        return loss_train, loss_val

      loss_train, loss_val = f3(x_train, layers, activation_function, y_train, ac, no_of_classes, loss_func, lambd, x_test, y_test)
      loss_arr.append(loss_val)
      accur_train = self.test_accuracy(layers, x_train, y_train, activation_function)
      accur_val = self.test_accuracy(layers, x_test, y_test, activation_function)



      name = "MOMENTUM"
      self.printingFunction(loss_train, loss_val, accur_train, accur_val, ep, name)

      


      if do_wandb_log == True:
        wandb_plots = dict({"epoch": ep+1 , "train_accuracy":accur_train,"train_error":loss_train,"val_accuracy":accur_val,"val_error":loss_val})
        wandb.log(wandb_plots)
      ep += 1
    return ac[len(ac)-1], y_test



  def nesterov_gradient_descent(self, x_train, y_train, x_test, y_test, no_of_classes, layers, activation_function, batch_size, eta, epochs, beta, loss_func,lambd, do_wandb_log):
    l = len(layers)
    prev_w, prev_b, loss_arr = [],[],[]
    i = 0
    while i < l-1:
      prev_w.append(np.zeros(self.W[i].shape))
      prev_b.append(np.zeros(self.B[i].shape))
      i += 1
    x_batch, y_batch = batch_converter(x_train, y_train, batch_size)

    ep = 0
    while ep < epochs:
      j = 0
      while j < len(x_batch):
        xb = x_batch[j]
        yb = y_batch[j]
        i = 0
        while i < l-1:
          self.W[i] += -beta * prev_w[i]
          self.B[i] += -beta * prev_b[i]
          i += 1
        preac, ac = self.forward(xb, layers, activation_function)
        grad_w, grad_b = self.backward(layers, xb, yb, no_of_classes, preac, ac, activation_function, loss_func)
        # print("grad_w", grad_w)
        i = 0
        while i < l-1:
          prev_w[i] = beta * prev_w[i] + grad_w[l-i-2]
          prev_b[i] = beta * prev_b[i] + grad_b[l-i-2]
          self.W[i] += -(eta * prev_w[i] + eta * lambd * self.W[i])
          self.B[i] += -eta * prev_b[i]
          i += 1
        j += 1


      loss_train = 0
      loss_val = 0

      def f4(x_train, layers, activation_function, y_train, ac, no_of_classes, loss_func, lambd, x_test, y_test):

        preac, ac = self.forward(x_train, layers, activation_function)
        loss_train = self.loss_function(y_train, ac[len(ac)-1], no_of_classes, loss_func, lambd)

        preac, ac = self.forward(x_test, layers, activation_function)
        loss_val = self.loss_function(y_test, ac[len(ac)-1], no_of_classes, loss_func, lambd)
        return loss_train, loss_val

      loss_train, loss_val = f4(x_train, layers, activation_function, y_train, ac, no_of_classes, loss_func, lambd, x_test, y_test)
      loss_arr.append(loss_val)
      accur_train = self.test_accuracy(layers, x_train, y_train, activation_function)
      accur_val = self.test_accuracy(layers, x_test, y_test, activation_function)



      name = "NAG"
      self.printingFunction(loss_train, loss_val, accur_train, accur_val, ep, name)

      



      if do_wandb_log == True:
        wandb_plots = dict({"epoch": ep+1 , "train_accuracy":accur_train,"train_error":loss_train,"val_accuracy":accur_val,"val_error":loss_val})
        wandb.log(wandb_plots)
      ep += 1
    return ac[len(ac)-1], y_test


  def rmsprop_gradient_descent(self, x_train, y_train, x_test, y_test, no_of_classes, layers, activation_function, batch_size, eta, epochs, beta, loss_func,lambd, do_wandb_log):
    eps = 1e-4
    vw, vb, loss_arr = [],[],[]
    l = len(layers)
    i = 0
    while i < l-1:
      vw.append(np.zeros(self.W[i].shape))
      vb.append(np.zeros(self.B[i].shape))
      i += 1

    x_batch, y_batch = batch_converter(x_train, y_train, batch_size)
    ep = 0
    while ep < epochs:
      j = 0
      while j < len(x_batch):
        xb = x_batch[j]
        yb = y_batch[j]
        preac, ac = self.forward(xb, layers, activation_function)
        grad_w, grad_b = self.backward(layers, xb, yb, no_of_classes, preac, ac, activation_function, loss_func)
        i = 0
        while i < l-2:
          vw[i] = beta * vw[i] + (1-beta) * grad_w[l-i-2] * grad_w[l-i-2]
          vb[i] = beta * vb[i] + (1-beta) * grad_b[l-i-2] * grad_b[l-i-2]
          self.W[i] =  self.W[i] - (eta * grad_w[l-i-2])/np.sqrt(vw[i] + eps) - (eta * lambd * self.W[i])
          self.B[i] =  self.B[i] - (eta * grad_b[l-i-2])/np.sqrt(vb[i] + eps)
          i += 1
        vw[i] = beta * vw[i] + (1-beta) * grad_w[l-i-2] * grad_w[l-i-2]
        vb[i] = beta * vb[i] + (1-beta) * grad_b[l-i-2] * grad_b[l-i-2]
        self.W[i] =  self.W[i] - (eta * grad_w[l-i-2])/np.sqrt(vw[i] + eps) - (eta * lambd * self.W[i])
        self.B[i] =  self.B[i] - (eta * grad_b[l-i-2])/np.sqrt(vb[i] + eps)
        j += 1


      loss_train = 0
      loss_val = 0

      def f5(x_train, layers, activation_function, y_train, ac, no_of_classes, loss_func, lambd, x_test, y_test):

        preac, ac = self.forward(x_train, layers, activation_function)
        loss_train = self.loss_function(y_train, ac[len(ac)-1], no_of_classes, loss_func, lambd)

        preac, ac = self.forward(x_test, layers, activation_function)
        loss_val = self.loss_function(y_test, ac[len(ac)-1], no_of_classes, loss_func, lambd)
        return loss_train, loss_val

      loss_train, loss_val = f5(x_train, layers, activation_function, y_train, ac, no_of_classes, loss_func, lambd, x_test, y_test)
      loss_arr.append(loss_val)
      accur_train = self.test_accuracy(layers, x_train, y_train, activation_function)
      accur_val = self.test_accuracy(layers, x_test, y_test, activation_function)

      
      name = "RMSPROP"
      self.printingFunction(loss_train, loss_val, accur_train, accur_val, ep, name)


      if do_wandb_log == True:
        wandb_plots = dict({"epoch": ep+1 , "train_accuracy":accur_train,"train_error":loss_train,"val_accuracy":accur_val,"val_error":loss_val})
        wandb.log(wandb_plots)
      ep += 1
    return ac[len(ac)-1], y_test




  def adam_gradient_descent(self, x_train, y_train, x_test, y_test, no_of_classes, layers, activation_function, batch_size, eta, epochs, beta1, beta2, loss_func,lambd, do_wandb_log):
    l = len(layers)
    mw, mb, vw, vb = [],[],[],[]
    eps = 1e-10
    loss_arr = []
    i = 0
    while i < l-1:
      vb.append(np.zeros(self.B[i].shape))
      mw.append(np.zeros(self.W[i].shape))
      vw.append(np.zeros(self.W[i].shape))
      mb.append(np.zeros(self.B[i].shape))
      i += 1
    x_batch, y_batch = batch_converter(x_train, y_train, batch_size)
    ep = 0
    while ep < epochs:
      j = 0
      while j < len(x_batch):
        xb = x_batch[j]
        yb = y_batch[j]
        preac, ac = self.forward(xb, layers, activation_function)
        grad_w, grad_b = self.backward(layers, xb, yb, no_of_classes, preac, ac, activation_function, loss_func)
        i = 0
        while i < l-1:
          temp = l-i-2
          mw[i] = beta1 * mw[i] + (1-beta1) * grad_w[temp]
          mb[i] = beta1 * mb[i] + (1-beta1) * grad_b[temp]
          mw_hat = mw[i] / (1 - np.power(beta1, j+1))
          mb_hat = mb[i] / (1 - np.power(beta1, j+1))

          vw[i] = beta2 * vw[i] + (1-beta2) * grad_w[temp] * grad_w[temp]
          vb[i] = beta2 * vb[i] + (1-beta2) * grad_b[temp] * grad_b[temp]
          vw_hat = vw[i] / (1 - np.power(beta2, j+1))
          vb_hat = vb[i] / (1 - np.power(beta2, j+1))

          self.W[i] = self.W[i] - (eta * mw_hat)/(np.sqrt(vw_hat) + eps)- (eta * lambd * self.W[i])
          self.B[i] = self.B[i] - (eta * mb_hat)/(np.sqrt(vb_hat) + eps)
          i += 1
        j += 1
      preac, ac = self.forward(x_train, layers, activation_function)
      loss_train = self.loss_function(y_train, ac[len(ac)-1], no_of_classes, loss_func, lambd)

      preac, ac = self.forward(x_test, layers, activation_function)
      loss_val = self.loss_function(y_test, ac[len(ac)-1], no_of_classes, loss_func, lambd)

      loss_arr.append(loss_val)
      accur_train = self.test_accuracy(layers, x_train, y_train, activation_function)
      accur_val = self.test_accuracy(layers, x_test, y_test, activation_function)

      


      name = "ADAM"
      self.printingFunction(loss_train, loss_val, accur_train, accur_val, ep, name)


      if do_wandb_log == True:
        wandb_plots = dict({"epoch": ep+1 , "train_accuracy":accur_train,"train_error":loss_train,"val_accuracy":accur_val,"val_error":loss_val})
        wandb.log(wandb_plots)
      ep += 1
    return ac[len(ac)-1], y_test


  def nadam_gradient_descent(self, x_train, y_train, x_test, y_test, no_of_classes, layers, activation_function, batch_size, eta, epochs, beta1, beta2, loss_func,lambd, do_wandb_log):
    eps = 1e-10
    mw, mb, vw, vb = [],[],[],[]
    x_batch, y_batch = batch_converter(x_train, y_train, batch_size)
    l = len(layers)
    loss_arr = []
    i = 0
    while i < l-2:
      vw.append(np.zeros(self.W[i].shape))
      vb.append(np.zeros(self.B[i].shape))
      mw.append(np.zeros(self.W[i].shape))
      mb.append(np.zeros(self.B[i].shape))
      i += 1
    vw.append(np.zeros(self.W[i].shape))
    vb.append(np.zeros(self.B[i].shape))
    mw.append(np.zeros(self.W[i].shape))
    mb.append(np.zeros(self.B[i].shape))
    x_batch, y_batch = batch_converter(x_train, y_train, batch_size)
    ep = 0
    while ep < epochs:
      j = 0
      while j < len(x_batch):
        xb = x_batch[j]
        yb = y_batch[j]
        preac, ac = self.forward(xb, layers, activation_function)
        grad_w, grad_b = self.backward(layers, xb, yb, no_of_classes, preac, ac, activation_function, loss_func)
        i = 0
        while i < l-2:
          mw[i] = beta1 * mw[i] + (1-beta1)* grad_w[l-i-2]
          mb[i] = beta1 * mb[i] + (1-beta1)* grad_b[l-i-2]
          mw_hat = mw[i] / (1 - np.power(beta1, j+1))
          mb_hat = mb[i] / (1 - np.power(beta1, j+1))

          vw[i] = beta2 * vw[i] + (1-beta2) * grad_w[l-i-2] * grad_w[l-i-2]
          vb[i] = beta2 * vb[i] + (1-beta2) * grad_b[l-i-2] * grad_b[l-i-2]
          vw_hat = vw[i] / (1 - np.power(beta2, j+1))
          vb_hat = vb[i] / (1 - np.power(beta2, j+1))

          self.W[i] = self.W[i] - (eta/(np.sqrt(vw[i]) + eps)) * (beta1 * mw_hat + (((1-beta1) * grad_w[l-i-2]) / (1 - np.power(beta1, j+1)))) - (eta * lambd * self.W[i])
          self.B[i] = self.B[i] - (eta/(np.sqrt(vb[i]) + eps)) * (beta1 * mb_hat + (((1-beta1) * grad_b[l-i-2]) / (1 - np.power(beta1, j+1))))
          i += 1
        mw[i] = beta1 * mw[i] + (1-beta1)* grad_w[l-i-2]
        mb[i] = beta1 * mb[i] + (1-beta1)* grad_b[l-i-2]
        mw_hat = mw[i] / (1 - np.power(beta1, j+1))
        mb_hat = mb[i] / (1 - np.power(beta1, j+1))

        vw[i] = beta2 * vw[i] + (1-beta2) * grad_w[l-i-2] * grad_w[l-i-2]
        vb[i] = beta2 * vb[i] + (1-beta2) * grad_b[l-i-2] * grad_b[l-i-2]
        vw_hat = vw[i] / (1 - np.power(beta2, j+1))
        vb_hat = vb[i] / (1 - np.power(beta2, j+1))

        self.W[i] = self.W[i] - (eta/(np.sqrt(vw[i]) + eps)) * (beta1 * mw_hat + (((1-beta1) * grad_w[l-i-2]) / (1 - np.power(beta1, j+1)))) - (eta * lambd * self.W[i])
        self.B[i] = self.B[i] - (eta/(np.sqrt(vb[i]) + eps)) * (beta1 * mb_hat + (((1-beta1) * grad_b[l-i-2]) / (1 - np.power(beta1, j+1))))
        j += 1

      loss_train = 0
      loss_val = 0

      def f5(x_train, layers, activation_function, y_train, ac, no_of_classes, loss_func, lambd, x_test, y_test):

        preac, ac = self.forward(x_train, layers, activation_function)
        loss_train = self.loss_function(y_train, ac[len(ac)-1], no_of_classes, loss_func, lambd)

        preac, ac = self.forward(x_test, layers, activation_function)
        loss_val = self.loss_function(y_test, ac[len(ac)-1], no_of_classes, loss_func, lambd)
        return loss_train, loss_val

      loss_train, loss_val = f5(x_train, layers, activation_function, y_train, ac, no_of_classes, loss_func, lambd, x_test, y_test)
      loss_arr.append(loss_val)
      accur_train = self.test_accuracy(layers, x_train, y_train, activation_function)
      accur_val = self.test_accuracy(layers, x_test, y_test, activation_function)

      
      name = "NADAM"
      self.printingFunction(loss_train, loss_val, accur_train, accur_val, ep, name)

      if do_wandb_log == True:
        wandb_plots = dict({"epoch": ep+1 , "train_accuracy":accur_train,"train_error":loss_train,"val_accuracy":accur_val,"val_error":loss_val})
        wandb.log(wandb_plots)
      ep += 1
    return ac[len(ac)-1], y_test


  def test_accuracy(self, layers1, x1, y1, activation_function1):
    layers, x, y, activation_function = layers1, x1, y1, activation_function1
    preac, ac = self.forward(x, layers, activation_function)
    y_pred = ac[len(ac)-1]
    err_count = 0
    i = 0
    while i < y_pred.shape[0]:
      maxval = -(math.inf)
      maxind = -1
      j = 0
      while j < y_pred.shape[1]:
        if maxval < y_pred[i][j]:
          maxval = y_pred[i][j]
          maxind = j
        j += 1
      if maxind != y[i]:
        err_count = err_count + 1
      i += 1
    temp = ((y.shape[0] - err_count)/y.shape[0])*100
    return temp
    
  def PlotError(self, ErrorSum):
    Iter = []
    for i in range(len(ErrorSum)):
      Iter.append(i)
    plt.plot(Iter,ErrorSum)
    plt.title('Error v/s Iteration')
    plt.xlabel('No of Iterations')
    plt.ylabel('Error')
    plt.show()


def main(x_train1, y_train1, x_val1, y_val1, input_size1, no_hidden_layers1, hidden_layer_size1, no_of_classes1, wt_initialisation1, optimiser1, activation_function1, batch_size1, eta1, epoch1, momentum1, beta1, beta11, beta21, loss_func1, lambd1, do_wandb_log1, plot_conf_mat1):
    com = optimiser1.upper() 
    layers = []
    layers.append(input_size1)
    i = 0
    while i < no_hidden_layers1:
      layers.append(hidden_layer_size1)
      i+=1
    layers.append(no_of_classes1)

    x_train, y_train, x_val, y_val, input_size, no_hidden_layers, hidden_layer_size, no_of_classes, wt_initialisation, optimiser, activation_function, batch_size, eta, epoch, momentum, beta, beta1, beta2, loss_func, lambd, do_wandb_log, plot_conf_mat = x_train1, y_train1, x_val1, y_val1, input_size1, no_hidden_layers1, hidden_layer_size1, no_of_classes1, wt_initialisation1, optimiser1, activation_function1, batch_size1, eta1, epoch1, momentum1, beta1, beta11, beta21, loss_func1, lambd1, do_wandb_log1, plot_conf_mat1

    object_neural_network = neural_network(layers, wt_initialisation)
    if com == "SGD":
      y_pred, y = object_neural_network.batch_grad_descent(x_train, y_train, x_val, y_val, no_of_classes, layers, activation_function, eta, batch_size, epoch, loss_func, lambd, do_wandb_log)
    
    if com == "MOMENTUM":
      y_pred, y = object_neural_network.momentum_grad_descent(x_train, y_train, x_val, y_val, no_of_classes, layers, activation_function, batch_size, eta, epoch, momentum, loss_func, lambd, do_wandb_log)
    if com == "NAG":
      y_pred, y = object_neural_network.nesterov_gradient_descent(x_train, y_train, x_val, y_val, no_of_classes, layers, activation_function, batch_size, eta, epoch, momentum, loss_func, lambd, do_wandb_log)

    if com == "RMSPROP":
      y_pred, y = object_neural_network.rmsprop_gradient_descent(x_train, y_train, x_val, y_val, no_of_classes, layers, activation_function, batch_size, eta, epoch, beta, loss_func, lambd, do_wandb_log)

    if com == "ADAM":
      y_pred, y = object_neural_network.adam_gradient_descent(x_train, y_train, x_val, y_val, no_of_classes, layers, activation_function, batch_size, eta, epoch, beta1, beta2, loss_func, lambd, do_wandb_log)

    if com == "NADAM":
      y_pred, y = object_neural_network.nadam_gradient_descent(x_train, y_train, x_val, y_val, no_of_classes, layers, activation_function, batch_size, eta, epoch, beta1, beta2, loss_func, lambd, do_wandb_log)

    if plot_conf_mat == True:
      y_pred = np.argmax(y_pred, axis = 1)
      print(y_pred.shape, y.shape)
      object_neural_network.Confunsion_Matrix_Plot(y_pred, y)

# main(x_train, y_train, x_val, y_val, input_size, no_hidden_layers, hidden_layer_size, no_of_classes, wt_initialisation, optimiser, activation_function, batch_size, eta, epoch, momentum, beta, beta1, beta2, loss_func, lambd, do_wandb_log, plot_conf_mat)


wandb.init(project = args.wandb_project , name = args.wandb_entity,entity = args.wandb_entity)

main(x_train, y_train, x_val, y_val, 784, args.num_layers, args.hidden_size, 10, args.weight_init, args.optimizer, args.activation, args.batch_size, args.learning_rate, args.epochs, args.momentum, args.beta, args.beta1, args.beta2, args.loss, args.weight_decay, True, False)

wandb.finish()