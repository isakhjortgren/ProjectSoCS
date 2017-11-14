import numpy as np


class Brain(object):
    def __init__(self, nbr_of_hidden_neurons, nbr_of_inputs=8, nbr_of_outputs=2):
        self.weights1 = np.random.rand(nbr_of_hidden_neurons, nbr_of_inputs + 1)
        self.weights2 = np.random.rand(nbr_of_outputs, nbr_of_hidden_neurons + 1)

    def make_decision(self, mean_prey_pos, mean_predator_pos, mean_prey_vel, mean_predator_vel):
        total_inputs = np.concatenate((mean_prey_pos, mean_predator_pos, mean_prey_vel, mean_predator_vel, [1]))
        hidden_state = np.tanh(np.dot(self.weights1, total_inputs))
        output_state = np.tanh(np.dot(self.weights2, hidden_state))
        norm = np.linalg.norm(output_state)
        if norm > 1:
            output_state = output_state/norm
        return output_state

