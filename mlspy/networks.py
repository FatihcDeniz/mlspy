import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

class Dense:
    def __init__(self,input_shape,output_shape,activation=None) -> None:
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weights = np.random.randn(input_shape,output_shape) / np.sqrt(input_shape + output_shape)
        self.bias = np.random.randn(1, output_shape) / np.sqrt(input_shape + output_shape)
    
    def forward(self,input_d) -> np.array:
        self.input =input_d
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self,output_error,learning_rate) -> np.array:
        #self.input = self.input.reshape(-1,1)
        if len(self.input.shape) == 1:
            self.input = self.input.reshape(1,-1)
        self.input_error = np.dot(output_error,self.weights.T)
        self.weights_error = np.dot(self.input.T, output_error)

        self.weights -= self.weights_error * learning_rate
        self.bias -= output_error * learning_rate
        
        return self.input_error

class ActivationLayer:
    def __init__(self,activation):
        self.activation = activation()
    
    def forward(self,X_train):
        self.input = X_train
        return self.activation(self.input)

    def backward(self,output_error,learning_rate=0.01):
        return self.activation.grad(self.input) * output_error
    
class Sequential:
    def __init__(self,loss) -> None:
        self.layers = []
        self.loss = loss()

    def add(self,layer):
        self.layers.append(layer)

    def predict(self,input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def fit(self,X_train,y_train,epochs = 1000,learning_rate = 0.1):
        self.error_all = []
        for i in range(epochs):
            err = 0
            for x,y in zip(X_train,y_train):
                output = x
                for layer in self.layers:
                    output = layer.forward(output)
                err += self.loss(y,output)
                
                error = self.loss.grad(y,output) 
                for layer in reversed(self.layers):
                    error = layer.backward(error,learning_rate)
            
            err /= X_train.shape[0]
            self.error_all.append(err)
            if len(self.error_all) > 30:
                print(np.round(np.mean(self.error_all[-5]),3), np.round(err,3) )
                if np.round(np.mean(self.error_all[-30]),3) < np.round(err,3) or err < 0.01:
                    break
            print(f"Epoch {i} --->", err)

    def plot_loss(self):
        plt.plot(range(0,len(self.error_all)),self.error_all)
        plt.show()

    def print_network(self):
        for layer in self.layers:
            print(layer.__class__.__name__,"\n")

class CNN: # Not finished yet
    def __init__(self, input_shape,kernel_size,depth) -> None:
        input_depth, input_height, input_width = input_shape
        self.depth =depth
        self.input_shape = input_shape
        self.input_depth = input_shape
        self.output_shape = (depth, input_height - kernel_size+1, input_width-kernel_size+1)
        self.kernels_shape = (depth,input_depth,kernel_size,kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.rand(*self.output_shape)

    def forward(self,input):
        self.input = input
        self.output = np.copy(self.biases)
