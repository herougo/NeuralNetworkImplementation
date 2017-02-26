import numpy as np

# we assume layers and dD's are vectors for simplicity

class Layer:
    def __init__(self, input):
        self.input = input

    def setup(self, input_len):
        raise NotImplementedError()

    def forward(self):
        raise NotImplementedError()

    def backward(self):
        raise NotImplementedError()

class FullyConnectedLayer(Layer):
    def __init__(self, output_len):
        self.output_len = output_len
        self.input_len = None
        self.W = None
        self.prev_X = None

    def setup(self, input_len):
    	self.input_len = input_len
        self.W = 2 * np.random.random((self.output_len, input_len)) - 1

    def forward(self, X):
        self.prev_X = X
        return np.dot(self.W, X).flatten()

    def backward(self, dD):
    	# TO DO: express dD and dW in a cleaner way
    	dD = np.transpose([dD])
        dW = np.dot(dD, np.array([self.prev_X]))
        assert dW.shape == self.W.shape

        dX = np.dot(np.transpose(self.W), dD).flatten()

        # update weights
        self.W += dW

        return dX

class SigmoidLayer(Layer):
    def __init__(self, output_len):
        self.output_len = output_len
        self.input_len = output_len
        self.prev_X = None

    def setup(self, input_len):
        if input_len != self.input_len:
            raise Exception("Sigmoid Layer doesn't match input lengths: %d -> %d" % (input_len, self.input_len))
        
    def _sigmoid(self, X):
        return 1 / (1 + np.exp(-1 * X))

    def _sigmoidDerivative(self, X):
        sig = self._sigmoid(X)
        return sig * (1 - sig)

    def forward(self, X):
        self.prev_X = X
        return self._sigmoid(X)

    def backward(self, dD):
    	# elementwise multiplication *
        return dD * self._sigmoidDerivative(self.prev_X)

class NeuralNetwork:
    def __init__(self, input_len):
        self.layers = []
        self.input_len = input_len
        self.output_len = input_len

    def forward(self, inputs):
        result = inputs
        for layer in self.layers:
            result = layer.forward(result)
        return result

    def backprop(self, inputs, outputs, iters=1000):
        inputs = np.array(inputs)
        outputs = np.array(outputs)

        for iter in range(iters):
            error_sum = 0
            for input, output in zip(inputs, outputs):
                # forward pass
                prev_output = input
                for i, layer in enumerate(self.layers):
                    prev_output = layer.forward(prev_output)

                # backward pass
                delta = np.sum(output - prev_output)
                prev_dD = delta
                for i in reversed(range(len(self.layers))):
                    layer = self.layers[i]
                    prev_dD = layer.backward(prev_dD)

                error_sum -= output * np.log(prev_output) + (1 - output) * np.log(1 - prev_output)

            if (iter + 1) % 100 == 0:
                print "Iteration", iter + 1, "Error:", error_sum

    def add(self, layer):
        layer.setup(self.output_len)
        self.layers.append(layer)
        self.output_len = layer.output_len


if __name__ == "__main__":
	np.random.seed(1)

    # "number of 1's is odd" examples
    X = [[0, 0, 0], # 0
         [0, 0, 1], # 1
         [0, 1, 0], # 1
         [0, 1, 1], # 0
         [1, 0, 0], # 1
         [1, 0, 1], # 0
         [1, 1, 0], # 0
         [1, 1, 1]] # 1

    Y = np.transpose([[0, 1, 1, 0, 1, 0, 0, 1]])

    # costs (after 10000 iterations) of various architectures:
    # 3 - 1			5.95506773
    # 3 - 3 - 1		3.28171725
    # 3 - 5 - 1		0.07159555
    # 3 - 7 - 1		0.7760022
    # 3 - 20 - 1	0.07159555
    # 3 - 7 - 3 - 1	0.76210421
    # 3 - 7 - 7 - 1	0.0330831

    nn = NeuralNetwork(input_len = 3)
    nn.add(FullyConnectedLayer(7))
    nn.add(SigmoidLayer(7))
    nn.add(FullyConnectedLayer(7))
    nn.add(SigmoidLayer(7))
    nn.add(FullyConnectedLayer(1))
    nn.add(SigmoidLayer(1))

    nn.backprop(X, Y, iters=10000)

    print "Results from backpropogation:"
    for x, y in zip(X, Y):
        print "  NN Output, Expected Output:", nn.forward(x), y
