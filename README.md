# Create-a-simple-neural-network-from-scratch
Learn to implement your own simple, vectorized neural network from scratch in python using numpy

![Neural Network Overview](https://matthewmazur.files.wordpress.com/2018/03/neural_network-9.png)  

This project aims to help you create a simple 3 layer neural network (1 output layer, 1 input layer, 1 hidden layer) with 2 activation neurons each by walking through how to implement forward and backpropagation in a vectorized format.   
  
- The sample neural network we will be building is based off of the following step-by-step backpropagation guide, which I highly recommend you begin with as it provides an excellent in-depth intuition for how each part of backpropagation is computed and will help you understand the vectorization a bit better.

> https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/.  




## Getting Started

### Weights  
The weights for each layer are stored in a 2x2 matrix, where each row indicates the neuron as a whole, and each column within the row is an individual weight attatched to that neuron. 
  
----------- Layer 1 --------------------- Layer 2 -----------  
![Layer1_weights](/Images/layer1_weights.png) ![Layer1_weights](/Images/layer2_weights.png)

### Layers  
Each computed layer will be stored in a 2x1 column vector 
  
--- Layer 1 ------ Layer 2 ------ Layer 3 ---  
![Layer1](/Images/Layer1.png) ![Layer2](/Images/layer2.png) ![Layer3](/Images/layer3.png)
  

## Forward Propagate

To get started we need to first calculate the values for each layer to be later used for back propagation.  

1. To find the value of a node first multiply each weight 'attatched' to the node, by a node in the previous layer, and taking the sum of all these values (The weighted sum) this value will be refered to as the 'Z' value. Note that each node has a weight for all the nodes in the previous layer.

 <p align="center">
  <img src="/Images/z_calculation.png">
</p>

  
2. Plug the calculated Z value into an activation function. The activation function we will be using for each node is the [Sigmoid Activation Function](https://en.wikipedia.org/wiki/Sigmoid_function). 
  
<p align="center">
  <img src="/Images/activation_calculation.png">
</p>
  
  
   If you are not yet familiar with how activation functions work I highly recommend watching this video on the topic.
   
<a align="center">
  <a href="http://www.youtube.com/watch?feature=player_embedded&v=m0pIlLfpXWE
  " target="_blank"><img src="http://img.youtube.com/vi/m0pIlLfpXWE/0.jpg"
  alt="Activation Function Video" width="220" height="160" border="10" /></a>

### Vectorized Implementation
Using vectorization we can caluculate the activation values for each layer in just one step, starting with the first hidden layer.

<p align="center">
  <img src="/Images/FFVectorized.png">
</p>



