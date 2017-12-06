import pickle
from PSO_class import PSO
from Aquarium import aquarium
import numpy as np


aquarium_parameters = {'nbr_of_prey': 15, 'nbr_of_pred': 2, 'size_X': 2, 'size_Y': 2, 'max_speed_prey': 0.1,
                       'max_speed_pred': 0.2, 'max_acc_prey': 1.0, 'max_acc_pred': 0.7, 'eat_radius': 0.05,
                       'weight_range': 1, 'nbr_of_hidden_neurons': 5, 'nbr_of_outputs': 2,
                       'visibility_range': 0.5, 'rand_walk_brain_set': [],
                       'input_set': ["enemy_pos", "friend_pos", "wall"], 'safe_boundary': False}

list_of_pso_prey = list()
list_of_pso_pred = list()

prey_iterations = 20
pred_iterations = 20

total_iteraions = 400
nbr_of_training_alternations = 2 * total_iteraions // (pred_iterations + prey_iterations)


try:
    for i in range(nbr_of_training_alternations):
        print('Training preys, iteration: ', i+1, ' out of ', nbr_of_training_alternations)
        # Train prey
        if i != 0:
            pso_prey = PSO(aquarium_parameters=aquarium_parameters, train_prey=True)
            pso_prey.nbr_of_iterations = prey_iterations
            for i_aquarium in pso_prey.list_of_aquarium:
                i_aquarium.pred_brain.update_brain(best_pred_brain)
            for i_aquarium in pso_prey.list_of_validation_aquarium:
                i_aquarium.pred_brain.update_brain(best_pred_brain)

            pso_prey.particle_best_position = list_of_pso_prey[-1].particle_best_position
            pso_prey.positions_matrix = list_of_pso_prey[-1].particle_best_position
            pso_prey.particle_best_value = pso_prey.run_all_particles()
            pso_prey.swarm_best_position = pso_prey.positions_matrix[np.argmax(pso_prey.particle_best_value), :]

            pso_prey.positions_matrix = list_of_pso_prey[-1].positions_matrix
            pso_prey.velocity_matrix = list_of_pso_prey[-1].velocity_matrix
            pso_prey.inertia_weight = list_of_pso_prey[-1].inertia_weight

        else:
            aquarium_parameters['rand_walk_brain_set'] = ['pred']
            pso_prey = PSO(aquarium_parameters=aquarium_parameters, train_prey=True)
            pso_prey.nbr_of_iterations = prey_iterations

        pso_prey.run_pso()
        aquarium_parameters['rand_walk_brain_set'] = []
        best_prey_brain = pso_prey.swarm_best_position

        list_of_pso_prey.append(pso_prey)

        print('Training predators, iteration: ', i+1, ' out of ', nbr_of_training_alternations)
        # Train predator
        pso_pred = PSO(aquarium_parameters=aquarium_parameters, train_prey=False)
        pso_pred.nbr_of_iterations = pred_iterations
        for i_aquarium in pso_pred.list_of_aquarium:
            i_aquarium.prey_brain.update_brain(best_prey_brain)
        for i_aquarium in pso_pred.list_of_validation_aquarium:
            i_aquarium.prey_brain.update_brain(best_prey_brain)

        if len(list_of_pso_pred) > 0:
            pso_pred.particle_best_position = list_of_pso_pred[-1].particle_best_position
            pso_pred.positions_matrix = list_of_pso_pred[-1].particle_best_position
            pso_pred.particle_best_value = pso_pred.run_all_particles()
            pso_pred.swarm_best_position = pso_pred.positions_matrix[np.argmax(pso_pred.particle_best_value), :]

            pso_pred.positions_matrix = list_of_pso_pred[-1].positions_matrix
            pso_pred.velocity_matrix = list_of_pso_pred[-1].velocity_matrix
            pso_pred.inertia_weight = list_of_pso_pred[-1].inertia_weight


        pso_pred.run_pso()
        best_pred_brain = pso_pred.swarm_best_position

        list_of_pso_pred.append(pso_pred)
except KeyboardInterrupt:
    print('Training aborted')

pso_data = {'list_of_pso_prey': list_of_pso_prey,
            'list_of_pso_pred': list_of_pso_pred}

with open('TrainingData.p', 'wb') as f:
    pickle.dump(pso_data, f)

