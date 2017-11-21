import numpy as np
import numpy.matlib as npml
from Aquarium import aquarium

class PSO(object):

    def __init__(self, train_prey=False):
        # aquarium parameters
        self.nbr_of_aquariums = 1
        self.nbr_of_hidden_neurons = 10
        self.nbr_of_inputs = 10
        self.nbr_of_outputs = 2
        self.weight_range = 5

        self.aquarium_parameters = {'nbr_of_prey': 15, 'nbr_of_pred': 2, 'size_X': 1, 'size_Y': 1,
                                   'max_speed_prey': 0.07, 'max_speed_pred': 0.1, 'max_acc_prey': 0.1,
                                   'max_acc_pred': 0.1, 'eat_radius': 0.1, 'weight_range': 5,
                                   'nbr_of_hidden_neurons': 10, 'nbr_of_inputs': 10, 'nbr_of_outputs': 2,
                                   'visibility_range': 0.3}

        self.aquarium_parameters_old = {'nbr_of_prey': 10, 'nbr_of_pred': 2, 'size_X': 1, 'size_Y': 1,
                                    'max_speed_prey': 0.01, 'max_speed_pred': 0.1, 'nbr_of_iterations': 100,
                                    'maximum_acceleration': 1, 'nbr_of_hidden_neurons': self.nbr_of_hidden_neurons,
                                    'nbr_of_inputs': self.nbr_of_inputs, 'nbr_of_outputs': self.nbr_of_outputs,
                                    'weight_range': self.weight_range}

        self.train_prey = train_prey

        # PSO parameters
        self.nbr_of_particles = 30
        self.nbr_of_iterations = 30
        self.maximum_velocity = 5
        self.c1 = 2
        self.c2 = 2
        self.inertia_weight = 1.4
        self.inertia_weight_lower_bound = 0.3
        self.beta = 0.99

        self.vector_length = (self.nbr_of_inputs + 1) * self.nbr_of_hidden_neurons + (self.nbr_of_hidden_neurons + 1) * self.nbr_of_outputs
        self.positions_matrix = self.weight_range * (2 * np.random.rand(self.nbr_of_particles, self.vector_length) - 1)
        self.velocity_matrix = self.maximum_velocity * (2 * np.random.rand(self.nbr_of_particles,
                                                                           self.vector_length) - 1)
        self.swarm_best_value = -np.inf
        self.swarm_best_position = None
        self.particle_best_value = np.zeros(self.nbr_of_particles)
        self.particle_best_position = np.copy(self.positions_matrix)
        self.list_of_swarm_best_value = list()
        self.list_of_aquarium = list()
        self.create_aquariums(self.nbr_of_aquariums)

    def create_aquariums(self, nbr_of_aquariums):
        self.nbr_of_aquariums = nbr_of_aquariums
        self.list_of_aquarium = list()
        for i in range(self.nbr_of_aquariums):
            self.list_of_aquarium.append(aquarium(**self.aquarium_parameters))

    def update_brain(self, pred_or_prey, array):
        if pred_or_prey == 'prey':
            for i_aquarium in self.list_of_aquarium:
                i_aquarium.prey_brain.update_brain(array)
        elif pred_or_prey == 'pred':
            for i_aquarium in self.list_of_aquarium:
                i_aquarium.pred_brain.update_brain(array)

    def run_pso(self):
        for i_iteration in range(self.nbr_of_iterations):
            print(f'Epoch number {i_iteration} out of {self.nbr_of_iterations}')

            particle_values = np.zeros(self.nbr_of_particles)
            for i_particle in range(self.nbr_of_particles):
                array = self.positions_matrix[i_particle, :]
                list_of_result = list()
                for i_aquarium in self.list_of_aquarium:
                    if self.train_prey:
                        i_aquarium.prey_brain.update_brain(array)
                    else:
                        i_aquarium.pred_brain.update_brain(array)

                    result_prey, result_pred = i_aquarium.run_simulation()

                    if self.train_prey:
                        list_of_result.append(result_prey)
                    else:
                        list_of_result.append(result_pred)

                particle_values[i_particle] = np.mean(list_of_result)

            iteration_best = np.max(particle_values)
            if iteration_best > self.swarm_best_value:
                self.swarm_best_value = iteration_best
                self.swarm_best_position = self.positions_matrix[np.argmax(particle_values), :]
            self.list_of_swarm_best_value.append(self.swarm_best_value)

            temp = particle_values > self.particle_best_value
            self.particle_best_value[temp] = particle_values[temp]
            self.particle_best_position[temp] = self.positions_matrix[temp]

            q = np.random.rand(self.nbr_of_particles, 1)
            cognitive_component = self.c1 * q * (self.particle_best_position - self.positions_matrix)

            best_global_position_matrix = npml.repmat(self.swarm_best_position, self.nbr_of_particles, 1)
            r = np.random.rand(self.nbr_of_particles, 1)
            social_component = self.c2 * r * (best_global_position_matrix - self.positions_matrix)

            new_particle_velocities = self.inertia_weight*self.velocity_matrix + cognitive_component + social_component

            self.positions_matrix = self.positions_matrix + new_particle_velocities
            self.velocity_matrix = new_particle_velocities
            self.velocity_matrix[self.velocity_matrix > self.maximum_velocity] = self.maximum_velocity
            self.velocity_matrix[self.velocity_matrix < -self.maximum_velocity] = -self.maximum_velocity

            if self.inertia_weight > self.inertia_weight_lower_bound:
                self.inertia_weight = self.inertia_weight * self.beta




