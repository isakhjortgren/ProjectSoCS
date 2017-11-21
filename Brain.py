import numpy as np


class Brain(object):
    def __init__(self, nbr_of_hidden_neurons, nbr_of_inputs=10, nbr_of_outputs=2,
                 weight_range=5):
        self.nbr_of_hidden_neurons=nbr_of_hidden_neurons
        self.nbr_of_inputs=nbr_of_inputs
        self.nbr_of_outputs=nbr_of_outputs
        self.weights1 = weight_range*(2*np.random.rand(nbr_of_hidden_neurons, nbr_of_inputs + 1) - 1)
        self.weights2 = weight_range*(2*np.random.rand(nbr_of_outputs, nbr_of_hidden_neurons + 1) - 1)

    def make_decision(self, total_inputs):
        total_inputs = np.concatenate((total_inputs, [1]))
        hidden_state = np.tanh(np.dot(self.weights1, total_inputs))
        hidden_state = np.concatenate((hidden_state, [1]))
        output_state = np.tanh(np.dot(self.weights2, hidden_state))
        norm = np.linalg.norm(output_state)
        if norm > 1:
            output_state = output_state / norm
        return output_state

    def update_brain(self, weight_array):
        number_weights1 = self.nbr_of_hidden_neurons*(self.nbr_of_inputs + 1)
        self.weights1 = weight_array[0: number_weights1].reshape(self.nbr_of_hidden_neurons, self.nbr_of_inputs + 1)
        self.weights2 = weight_array[number_weights1:].reshape(self.nbr_of_outputs, self.nbr_of_hidden_neurons + 1)
#ToDo Add function that sets weight with given input weights vector


class randomBrain(Brain):
    def make_decision(self, total_inputs):
        return (np.random.random((1,2))-0.5)*2

