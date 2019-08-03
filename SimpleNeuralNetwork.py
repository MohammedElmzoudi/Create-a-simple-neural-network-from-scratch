import numpy as np
from tqdm import tqdm
# A 3 layer neural network with 2 neurons each


#weights = (number of neurons in current layer)
#          x(number of neurons in previous layer)
class NeuralNetwork:
    def __init__(self):
        # Starting weight values
        self.weights1 = np.array([[0.150000,0.200000],[0.250000,0.30000]])
        self.weights2 = np.array([[0.400000,0.450000],[0.500000,0.55000]])
        # Starting bias values
        self.b1 = 0.35
        self.b2 = 0.60
        # Input values and desired output values
        self.input_values = np.array([0.050000, 0.100000])
        self.output_values = np.array([0.010000, 0.99000])
        # Turning input and output values into column vectors
        self.input_values = np.expand_dims(self.input_values, axis=1)
        self.output_values = np.expand_dims(self.output_values, axis=1)

    def train(self, a):
        #Forward Propagate

        ## Calculate z/input values and activation values for each layer
        z_hidden_layer1 = np.matmul(self.weights1,self.input_values) + self.b1
        hidden_layer1 = self.sigmoid(z_hidden_layer1)
        z_output_layer = np.matmul(self.weights2,hidden_layer1) + self.b2
        output_layer = self.sigmoid(z_output_layer)

        #Backpropagate

        ## Calculate delta values
        output_layer_delta = (output_layer - self.output_values) * self.sigmoid_derivative(z_output_layer)
        #output_layer_delta = (output_layer - self.output_values)
        hidden_layer1_delta = np.matmul(self.weights2.T,output_layer_delta)
        hidden_layer1_delta = hidden_layer1_delta * self.sigmoid_derivative(z_hidden_layer1)

        ## Calculate final weight gradients by taking outer product with the previous layer
        weights2_gradient = np.outer(output_layer_delta,hidden_layer1.T)
        weights1_gradient = np.outer(hidden_layer1_delta,self.input_values.T)

        ## Update weights
        self.weights1 = self.weights1 - a * weights1_gradient
        self.weights2 = self .weights2 - a * weights2_gradient
        #return the updated theta values
        return self.weights1 , self.weights2


    def sigmoid(self, z):
        return 1 / (1+np.exp(-z))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1-self.sigmoid(z))

    #Returns the current error of the model
    def predict(self):
        z_hidden_layer1 = np.matmul(self.weights1, self.input_values) + self.b1
        hidden_layer1 = self.sigmoid(z_hidden_layer1)
        z_output_layer = np.matmul(self.weights2, hidden_layer1) + self.b2
        output_layer = self.sigmoid(z_output_layer)
        return output_layer

    def error(self):
        output_layer = self.predict()
        return sum((1/2) * (self.output_values - output_layer)**2)


if __name__ == "__main__":
    ann = NeuralNetwork()
    updated_weights1, updated_weights2 = 1,2
    for i in tqdm(range(10000)):
        updated_weights1, updated_weights2 = ann.train(0.5)

    print("New layer 1 weights: \n" ,updated_weights1)
    print(" New layer 2 weights: \n", updated_weights2)
    print("Error value:", ann.error())
    print("Prediction with new weights:", ann.predict())












        
