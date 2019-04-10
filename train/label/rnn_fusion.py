import numpy as np


class Rnn:

    def __init__(self):
        self.alpha = 0.1
        self.input_dim = 2
        self.hidden_dim = 16
        self.output_dim = 1

        self.synapse_0 = 2 * np.random.random((self.input_dim,self.hidden_dim)) - 1
        self.synapse_1 = 2 * np.random.random((self.hidden_dim, self.output_dim)) - 1
        self.synapse_h = 2 * np.random.random((self.hidden_dim, self.hidden_dim)) - 1

        self.synapse_0_update = np.zeros_like(self.synapse_0)
        self.synapse_1_update = np.zeros_like(self.synapse_1)
        self.synapse_h_update = np.zeros_like(self.synapse_h)

    def train(self):
        pass


def sigmoid(x):
        output = 1/(1+np.exp(-x))
        return output


def sigmoid_output_to_derivative(output):
        return output * (1 - output)


if __name__ == '__main__':
    pass
