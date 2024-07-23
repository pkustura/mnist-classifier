import numpy as np
import random

class Network(object):

    # sizes: an array of the size of each layer
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        # we need 1 bias for each neuron in each layer
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] 
        # we need (number of neurons in last layer) weights for each neuron in the layer
        #self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[:1])]
        print("initializing weights:")
        self.weights = []
        for i in range(len(sizes) - 1):
            input_size = sizes[i]
            output_size = sizes[i+1]
            # print(f"Creating weight matrix for layer {i+1}: shape ({output_size}, {input_size})")
            weight_matrix = np.random.randn(output_size, input_size)
            # print(f"Created weight matrix shape: {weight_matrix.shape}")
            self.weights.append(weight_matrix)

        print(f"Initialized network with sizes: {sizes}")
        print(f"Weights shapes: {[w.shape for w in self.weights]}")
        print(f"Biases shapes: {[b.shape for b in self.biases]}") 
    # a is input neuron activations.
    # it is assumed a is an (n,1) ndarray.
    # the dot basically does weighted sum of a with each row, returns (n,1) array.
    def feedforward(self, a):
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a


    # training_data is list of tuples (x,y) : inputs, desired outputs
    # if test_data is provided then evaluation will be done and partial progress printed (slows things down quite a bit)
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        print(f"Training data size: {len(training_data)}")
        print(f"First training example - Image shape: {training_data[0][0].shape}, Label shape: {training_data[0][1].shape}")
        if test_data: n_test = len(test_data)
        n = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {j} complete")
    
    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            x = np.array(x).reshape(-1, 1)  # Ensure x is a column vector
            y = np.array(y).reshape(-1, 1)  # Ensure y is a column vector
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]
        # eta/len(mini_batch) : eta is the learning rate. I guess with a smaller batch we move it MORE because the nabla has accumulated less.
        # makes sense.


    def backprop(self, x, y):
    
        # print(f"Network sizes: {self.sizes}")
        # print(f"Weights shapes: {[w.shape for w in self.weights]}")
        # print(f"Biases shapes: {[b.shape for b in self.biases]}")
        # print(f"Input x shape: {x.shape}")
        # print(f"Input y shape: {y.shape}")
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
    
        # feedforward
        activation = x
        activations = [x]
        zs = []
        # print(f"Input shape: {activation.shape}")
    
        for i, (b, w) in enumerate(zip(self.biases, self.weights)):
            # print(f"Layer {i+1} - Weight shape: {w.shape}, Bias shape: {b.shape}")
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
            # print(f"Layer {i+1} - Output shape: {activation.shape}")
    
        # print(f"Final activation shape: {activations[-1].shape}")
        # print(f"y shape: {y.shape}")
    
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
    
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
    
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(x == y for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

    # z is a vector/numpy array. np applies sigmoid elementwise
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_prime(z):
    return sigmoid(z) * (1-sigmoid(z))
