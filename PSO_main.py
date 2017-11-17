import numpy as np
import numpy.matlib as npml
from Aquarium import aquarium
from Brain import Brain

def RunSimulation(fish_array, shark_array=0):
    return 1/(np.linalg.norm(fish_array)+1)

nbr_of_particles = 30
nbr_of_iterations = 10000

nbr_of_hidden_neurons = 1
nbr_of_inputs = 2
nbr_of_outputs = 2
weight_range = 5
maximum_velocity = 5
c1 = 2
c2 = 2
inertia_weight = 1.4
inertia_weight_lower_bound = 0.3
beta = 0.99

aquarium_paramters = {'nbr_of_prey': 10, 'nbr_of_pred': 2, 'size_X': 1, 'size_Y': 1, 'max_speed_prey': 0.01,
                      'max_speed_pred': 0.1, 'nbr_of_iterations': 100, 'maximum_acceleration': 1}

vector_length = (nbr_of_inputs+1)*nbr_of_hidden_neurons + (nbr_of_hidden_neurons+1)*nbr_of_outputs

positions_matrix = weight_range * (2 * np.random.rand(nbr_of_particles, vector_length) - 1)
velocity_matrix = maximum_velocity * (2 * np.random.rand(nbr_of_particles, vector_length) - 1)

swarm_best_value = 0
particle_best_value = np.zeros(nbr_of_particles)
particle_best_position = np.copy(positions_matrix)
aquarium_1 = aquarium(**aquarium_paramters)

for i_iteration in range(nbr_of_iterations):

    particle_values = np.zeros(nbr_of_particles)
    for i_particle in range(nbr_of_particles):
        array = positions_matrix[i_particle, :]
        aquarium_1.prey_brain = array
        aquarium_1.pred_brain = pred_array
        particle_values[i_particle] = aquarium_1.run_simulation()

    iteration_best = np.max(particle_values)
    if iteration_best > swarm_best_value:
        swarm_best_value = iteration_best
        swarm_best_position = positions_matrix[np.argmax(particle_values), :]

    temp = particle_values > particle_best_value
    particle_best_value[temp] = particle_values[temp]
    particle_best_position[temp] = positions_matrix[temp]

    q = np.random.rand(nbr_of_particles, 1)
    cognitive_component = c1 * q * (particle_best_position - positions_matrix)

    best_global_position_matrix = npml.repmat(swarm_best_position, nbr_of_particles, 1)
    r = np.random.rand(nbr_of_particles, 1)
    social_component = c2 * r * (best_global_position_matrix - positions_matrix)

    new_particle_velocities = inertia_weight * velocity_matrix + cognitive_component + social_component

    positions_matrix = positions_matrix + new_particle_velocities
    velocity_matrix = new_particle_velocities
    velocity_matrix[velocity_matrix > maximum_velocity] = maximum_velocity
    velocity_matrix[velocity_matrix < -maximum_velocity] = -maximum_velocity

    if inertia_weight > inertia_weight_lower_bound:
        inertia_weight = inertia_weight * beta




print(swarm_best_position)










