import mnist_loader
import network

training_data, test_data = mnist_loader.structured_load()

net = network.Network([784,20,10])

net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
