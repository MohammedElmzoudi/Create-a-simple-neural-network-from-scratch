# Create-a-simple-neural-network-from-scratch
Learn to implement your own simple, vectorized neural network from scratch in python using numpy

![Neural Network Overview](https://matthewmazur.files.wordpress.com/2018/03/neural_network-9.png)  

This project aims to help you create a simple 3 layer neural network (1 output layer, 1 input layer, 1 hidden layer) with 2 activation neurons each by walking through how to implement forward and backpropagation in a vectorized format.   
  
- The sample neural network we will be building is based off of the following step-by-step backpropagation guide, which I highly recommend you begin with as it provides an excellent in-depth intuition for how each part of backpropagation is computed and will help you understand the vectorization a bit better.

> https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/.  




## Getting Started

### Weights  
The weights for each layer are stored in a 2x2 matrix, where each row indicates the neuron as a whole, and each column within the row is an individual weight attatched to that neuron.
   
![Layer1_weights](/Images/layer1_weights.png) ![Layer1_weights](/Images/layer2_weights.png)

### Layers  
Each computed layer will be stored in a 2x1 column vector  
>Layer 1 ------- Layer 2 ------- Layer 3
  
![Layer1](/Images/Layer1.png) ![Layer2](/Images/layer2.png) ![Layer3](/Images/layer3.png)
