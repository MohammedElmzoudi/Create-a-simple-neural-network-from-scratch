# Create-a-simple-neural-network-from-scratch
Learn how to create a fully functioning neural network with fully vectorized and simple implementations of forward and back propagation in python without the use of external machine learning libraries.

<p align="center">
  <img src="/Images/neural_network-diagram.png">
</p>  

This project aims to help you create a simple 3 layer neural network (1 output layer, 1 input layer, 1 hidden layer) with 2 activation neurons each by walking through how to implement forward and backpropagation in a vectorized format.   
  
- The sample neural network we will be building is based off of the following step-by-step backpropagation guide, which I highly recommend you begin with as it provides an excellent in-depth intuition for how each part of backpropagation is computed and will help you understand the vectorization a bit better.

> https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/.  

## Getting Started

### Weights  
The weights for each layer are stored in a 2x2 matrix, where each row indicates the neuron as a whole, and each column within the row is an individual weight attatched to that neuron.   
  
<p align="center">
  <img src="/Images/layer1_weights.png">
  <img src="/Images/layer2_weights.png">
</p>
  
> Weights for layers 2 and 3 respectively
   
### Layers  
Each computed layer will be stored in a 2x1 column vector 
  
<p align="center">
  <img src="/Images/Layer1.png">
  <img src="/Images/layer2.png">
  <img src="/Images/layer3.png">
</p>

> Layers 1, 2, and 3 respectively
  
  
## Forward Propagation

To get started we need to first calculate the values for each layer to be later used for back propagation.  

1. To find the value of a node first multiply each weight 'attatched' to the node, by a node in the previous layer, and taking the sum of all these values,(The weighted sum) this value will be refered to as the 'Z' value. Note that each node has a weight for all the nodes in the previous layer.

 <p align="center">
  <img src="/Images/z_calculation.png">
</p>

2. Plug the calculated Z value into an activation function. The activation function we will be using for each node is the [Sigmoid Activation Function](https://en.wikipedia.org/wiki/Sigmoid_function). 
  
<p align="center">
  <img src="/Images/activation_calculation.png">
</p>
   - If you are not yet familiar with how activation functions work I highly recommend watching this video on the topic.
<p align="center">
  <a href="http://www.youtube.com/watch?feature=player_embedded&v=m0pIlLfpXWE
  " target="_blank"><img src="http://img.youtube.com/vi/m0pIlLfpXWE/0.jpg"
  alt="Activation Function Video" width="220" height="160" border="10" /></a>

### Vectorized Implementation
Using vectorization we can caluculate the activation values for each layer in just one step, starting with the first hidden layer and continuing until you reach the output layer.

<p align="center">
  <i><sub> Layer 1 <br/>
  <img src="/Images/FFVectorized.png">
  <br/>
  <img src="/Images/FFVectorized1.png">
  <br/>
  <br/>
  Layer 2 <br/>
  <img src="/Images/FFVectorized2.png">
  <br/>
  <img src="/Images/FFVectorized3.png">
  <br/>
  <i/><sub/>
</p>


And in python:   
```
 z_hidden_layer1 = np.matmul(weights1,input_values) + bias1
 hidden_layer1 = sigmoid(z_hidden_layer1)
 
 z_output_layer = np.matmul(weights2,hidden_layer1) + bias2
 output_layer = sigmoid(z_output_layer)
```
## Backpropagation

### Brief Explanation
- Each machine learning algorithm has what is called a loss function. This function determines how well a model is performing on a certain dataset based on how close the values that the model outputs when given a set of inputs is to the actual desired value of those inputs. The goal of any machine learning model is to minimize this cost function by finding the optimal value for each weight.
- By taking the derivative of the cost function with respect to each weight, we can find the slope of the line on the cost function when that weight is a certain value. A positive slope indicates that if the value of the weight is decreased then the cost function will also decrease, and vise-versa. By continuously subtracting the derivative w.r.t thetea from theta itself, we can eventually hope to reach a global minimum of the cost function. You can learn more about the topic [here](https://en.wikipedia.org/wiki/Gradient_descent). 
 <p align="center">
  <img src="https://thumbs.gfycat.com/KindAmpleImperialeagle-size_restricted.gif">
</p>

### Notation

<p align="center">
  <b><a>Cost Function:</a></b>
  <br><br>
  <img src="/Images/CostN.png">
</p>




<p align="center">
  <b><a>Weights:</a></b>
  <br><br>
  <img src="/Images/ThetaN.png"><br/>  
  <i><a>l = the layer of the neurons that the weights effects.</a><br/>
  <i><a>i = The specific neuron that the weight is attatched to.</a><br/>
  <i><a>j = the neuron of the previous layer that is multiplied by the weight. </a><br/>
</p>

<p align="center">
  <b><a>Weight l,i,j gradient:</a></b>
  <br><br>
  <img src="/Images/NablaN.png"><br/>  
</p>

<p align="center">
  <b><a>Delta (Will go over this later on):</a></b>
  <br><br>
  <img src="/Images/deltaN.png"><br/>  
</p>





### Derivation / Proof

__Goal__ : Find Derivative of the cost function with respect to a specific weight l,i,j. (Note that this derivation works so that any cost function can be used).  
- Also note that knowledge of the [Chain Rule](https://www.youtube.com/watch?v=6kScLENCXLg&vl=en) is required as it is used heavily throughout this proof.   
  
<p align="center">
  <br/>
  <img src="/Images/deriv0.png">
  <br/>
</p>
  
1. Using the chain rule we know that this equation can simplify.  
<p align="center">
  <br/>
  <img src="/Images/deriv1.png">
  <br/>
</p>
  
2. Remember that the Z value is just the weighted sum of the neurons in the previous layer.  
<p align="center">
  <br/>
  <img src="/Images/deriv2.png"><br/>  
  <br/>
  <sub><i><a>When taking the derivative with respect to a variable, every term that does not contain the variable will equate to zero</a><br/>
  <i><a>Because of this, we can ignore all summations that do not contain theta. </a><br/>
  <i><a> 2.1 </a><br/>
  <br/>
  <img src="/Images/deriv2.1.png"><br/>
  <br/>
  <i><a>And we can treat z as a single, simple term and take its derivative. </a><br/>
  <br/>
  <i><a> 2.2 </a><br/>
  <br/>
  <img src="/Images/deriv2.2.png"><br/>
  <br/>
  <i><a> 2.3 </a><br/>
  <br/>
  <img src="/Images/deriv2.3.png"><br/>
  <br/>
  <i><a> 2.4 </a><br/>
  <br/>
  <img src="/Images/deriv2.4.png"><br/>
  <br/>
  <i><a>Now the derivative of z^l is just the neuron in the previous layer that multiplies the theta we are solving for. </a><br/></sub>
</p>
    
3. The derivative of the cost function with respect to any z value is called delta, or the error value for that node. Because 'z' is the weighted sum of all the nodes in the previous layer, if the derivative of the cost w.r.t 'z' (or the slope of the tangent line on the cost function when z is a certain value ) is extremely low, this means that your current z value is already in it's best position to provide the lowest possible value of the cost function and vise versa. This essentially tells you how 'wrong' your combination of weights and biases are, hence the name delta.
<p align="center">
  <sub><i><a><b> Goal: Find- </a><br/></b>
  <br/>
  <img src="/Images/deriv3.0.png"><br/>  
  <br/>
  <i><a> 3.1 </a><br/>
  <i><a> Using the chain rule we can break this down further </a><br/>
  <br/>
  <img src="/Images/deriv3.1.png"><br/>  
  <br/>
  <i><a>  </a> Intuitively this makes sense. Given a node, all the nodes in the layer after it take said node into their calculation, so it only makes sense that when trying to get to the z value of a node through the chain rule, you must go through the outermost layers first to find it's derivative, like peeling an oninon to get to it's center. <br/>
  <br/>
  <i><a> 3.2 </a><br/>
  <i><a> As pointed out earlier, the derivative of the cost function with respect to any Z value is called delta, so we can re-write this as: </a><br/>
  <br/>
  <img src="/Images/deriv3.2.png"><br/>  
  </sub>
</p>
    
 4. Now we just need to find the derivative of each z value in the next layer with respect to the z value of the current layer.
 <p align="center">
  <sub> <i><a> 4.1 </a><br/>
  <i><a> The Z values of any layer are just the weighted sum of the values in the previous layer, so the z values in layer l+1 should be the weighted sum of the output values of layer l </a><br/>
  </br>
  <img src="/Images/deriv4.1.png"><br/>  
  </br>
  <i><a> 4.2 </a><br/>
  <i><a> Similar to our earlier situation, any term that does not contain the z value we are finding the derivative with respect to, will equate to zero, so we can re-write the term as follows:</a><br/>
  </br>
  <img src="/Images/deriv4.2.png"><br/> 
  </br>
  <i><a> 4.3 </a><br/>
  <i><a> Which then becomes:</a><br/>
  </br>
  <img src="/Images/deriv4.3.png"><br/> 
  </br>
  <img src="/Images/deriv 4.35.png"><br/>   
  </sub>
</p>

5. We've done it! After this derivation we end up with:
    
<p align="center">
  <sub><br/><i><a> 5.1 </a><br/>
  <img src="/Images/deriv5.png"><br/>  
  <br/>
  <i><a> 5.2 </a><br/>
  <i><a> This entire term is the delta value of the node theta is attatched to: </a><br/>
  <br/>
  <img src="/Images/deriv5.22.png"><br/>  
  <br/>
  <i><a> 5.3 </a><br/>
  <i><a> So we can re-write this as: </a><br/>
  <br/>
  <img src="/Images/deriv5.3.png"><br/>  
  <br/>
  </sub>
</p>
    
- In total, what this really means is that we multiply the delta value of every node in the next layer, with the theta that multiplies the node connected with the theta value that we are finding the gradient for in the current layer (the 'current node'). After this we multiply by the derivative of our sigmoid function with respect to the current node's z value and finally multiply by the output value of the node in the previous layer that multiplies with the theta we are finding the gradient for to form the current node's z value. Wow, that was a mouthful. I really suggest taking a moment (or a few weeks) to sit down and really digest what is happening here to fully grasp this crucial concept in neural networks.
   
### Delta Values  
- The delta values extremely useful, because every theta gradient within a node uses the same delta, so by finding the delta value we already have a large piece of the gradient completed for each theta within the node.  
- Each delta value builds on the delta values in the next layer, so to find each delta we must start with the final layer.  
- Note that for each cost function the computation of the last layer's delta will be different, but once that is computed all the deltas in following layers build upon it the same way.  
- For this neural network, our cost function will be squared error:

<p align="center">
  <sub>
  <img src="/Images/SquaredError.png"><br/>  
  </br>
  <br/><i><a> Find: </a><br/>
  <img src="/Images/delta1.png"><br/>  
  <img src="/Images/delta2.png"><br/> 
  <img src="/Images/delta3.png"><br/> 
  <br/>
  <img src="/Images/delta4.1.png"><br/> 
  </br>
  <br/><i><a> After repeating this step for each delta value in the final layer we can now proceed to the next layer using these deltas. </a><br/>
  <img src="/Images/delta5.png"><br/> 
  <br/>
  <img src="/Images/delta6.png"><br/>  
  <img src="/Images/delta7.png"><br/> 
  </sub>
</p>
    
  
### Vectorized Implementation  
  
- __Goal:__ Find a way to quickly and efficiently implement backpropagation using vectorization.  
<p align="center">
  <img src="/Images/neural_network-diagram.png"><br/> 
  <br/>
</p>
    
1. The convenient thing about backpropagation, is that every theta gradient for each node has the same piece within it - the delta for that node. With this knowledge we can find the delta values for each layer first, then derive each theta gradient from those delta values.  
  
<p align="center">
  <sub><br/><i><a> We first find the delta values for the output layer. </a><br/>
  <img src="/Images/deltaV1.png"><br/>
  <br/><i><a> Note that '.*' is an element-wise operation, because we want to multiply each node by the derivative of the sigmoid function with respect to it's own z value. </a><br/>
  <br/>
  <img src="/Images/deltaV2.png"><br/>
  <br/>
  <img src="/Images/deltaV4.1.png"><br/>
  <img src="/Images/deltaV3.1.png"><br/>
  </sub>
</p>
     
 - In python:  
   
```
## Calculate delta values
output_layer_delta = (output_layer - self.output_values) * self.sigmoid_derivative(z_output_layer) 
hidden_layer1_delta = np.matmul(self.weights2.T,output_layer_delta)  
hidden_layer1_delta = hidden_layer1_delta * self.sigmoid_derivative(z_hidden_layer1)  
```
  
2. With our new delta values we can now find the gradient for each theta. Remember that we want each theta gradient value to be equal to the curent node's delta * the node in the previous layer that the weight is attached to.
- This can be accomplished by taking the outer product of the two vectors.
  
<p align="center">
  <br/>
  <sub><br/><i><a> We can start by finding the output layer's gradients: </a><br/>
  <br/>
  <img src="/Images/totalV1.png"><br/>
  <br/>
  <img src="/Images/totalV2.png"><br/>
  <img src="/Images/totalV3.1.png"><br/>
  <br/>
  <br/><i><a> Repeating for layer two we get: </a><br/>
  <br/> 
  <img src="/Images/totalV4.png"><br/>
  <br/>
  <img src="/Images/totalV5.png"><br/>
  <img src="/Images/totalV6.1.png"><br/>
  </sub>
</p>
      
 - In python:  
     
 ```
 ## Calculate final weight gradients by taking outer product with the previous layer
 weights2_gradient = np.outer(output_layer_delta,hidden_layer1.T)
 weights1_gradient = np.outer(hidden_layer1_delta,self.input_values.T)
 ```
      
 3. Now we multiply our gradient by an alpha value which determines the rate at which we descent, and subtract from our  original theta values.
 <p align="center">
  <br/>
  <sub><i> Output layer <br/>
  <img src="/Images/tUpdate3.png"><br/>
  <img src="/Images/tUpdate4.png"><br/>
  <br/>
  <br/>
  <i> layer 2 <br/>
  <img src="/Images/tUpdate1.png"><br/>
  <img src="/Images/tUpdate2.png"><br/>
  </sub>
</p>
      
- In python:  
    
```
## Update weights
self.weights1 = self.weights1 - a * weights1_gradient
self.weights2 = self.weights2 - a * weights2_gradient
```
 
- We have now sucessfully updated our theta values! After running 10,000 iterations our error decreases 99.9882%, and the model returns values very close to the desired output of [.01, 0.99], returning [0.00885999 0.9897063] as opposed to the starting value of [0.7478592  0.78256178].
- Note that this neural network currenly takes in only one training input, but if you were to take in 'm' training examples you would just need to divide the gradients by m, to get the average change for each theta.

## Resources
http://www.youtube.com/watch?feature=player_embedded&v=m0pIlLfpXWE
https://www.youtube.com/watch?v=6kScLENCXLg&vl=en
https://thumbs.gfycat.com/KindAmpleImperialeagle-size_restricted.gif
https://en.wikipedia.org/wiki/Gradient_descent
https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
https://en.wikipedia.org/wiki/Sigmoid_function
https://stats.stackexchange.com/questions/94387/how-to-derive-errors-in-neural-network-with-the-backpropagation-algorithm

All equations were created using online LaTeX equation editor
-http://rogercortesi.com/eqn/index.php

-Guide by Mohammed Elmzoudi



