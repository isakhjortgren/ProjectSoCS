import pickle
import os
from PSO_class import PSO
import traceback
from Aquarium import aquarium
import numpy as np


aquarium_parameters = {'nbr_of_prey': 20, 'nbr_of_pred': 3, 'size_X': 2, 'size_Y': 2, 'max_speed_prey': 0.2,
                       'max_speed_pred': 0.2, 'max_acc_prey': 0.2, 'max_acc_pred': 0.2, 'eat_radius': 0.05,
                       'weight_range': 0.5, 'nbr_of_hidden_neurons': 4, 'nbr_of_outputs': 2,
                       'visibility_range': 0.5, 'rand_walk_brain_set': [], 'input_type': 'closest',
                       'input_set': ["enemy_pos", "friend_pos", "wall"], 'safe_boundary': False}

list_of_pso_prey = list()
list_of_pso_pred = list()
start_iteration = 0
if os.path.isfile('TrainingData.p'):
    var = str(input("Continue training? (y/n): "))

    if var == 'y':
        print('Continue training!')
        with open('TrainingData.p', 'rb') as f:
            pso_data = pickle.load(f)
        list_of_pso_prey = pso_data['list_of_pso_prey']
        list_of_pso_pred = pso_data['list_of_pso_pred']
        start_iteration = len(list_of_pso_pred)
        aquarium_parameters = list_of_pso_pred[-1].aquarium_parameters
        print('current aquarium parameters:\n', aquarium_parameters)
        best_pred_brain = list_of_pso_pred[-1].swarm_best_position
        best_prey_brain = list_of_pso_prey[-1].swarm_best_position

        if start_iteration == 0:
            raise RuntimeError('No data in TrainingData.p file use new one!')
    elif var == 'n':
        print('Starting new training!')
    else:
        raise Exception('abort training, wrong input')

prey_iterations = 20
pred_iterations = 20

total_iteraions = 400
nbr_of_training_alternations = 2 * total_iteraions // (pred_iterations + prey_iterations)



try:
    for i in range(start_iteration, nbr_of_training_alternations):
        print('Training preys, alternation: ', i+1, ' out of ', nbr_of_training_alternations)
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
            pso_prey.particle_best_value = pso_prey.run_all_particles("prey")
            pso_prey.swarm_best_position = pso_prey.positions_matrix[np.argmax(pso_prey.particle_best_value), :]

            pso_prey.positions_matrix = list_of_pso_prey[-1].positions_matrix
            pso_prey.velocity_matrix = list_of_pso_prey[-1].velocity_matrix
            pso_prey.inertia_weight = list_of_pso_prey[-1].inertia_weight

        else:
            aquarium_parameters['rand_walk_brain_set'] = ['pred']
            pso_prey = PSO(aquarium_parameters=aquarium_parameters, train_prey=True)
            pso_prey.nbr_of_iterations = prey_iterations

        pso_prey.run_pso("prey")
        aquarium_parameters['rand_walk_brain_set'] = []
        best_prey_brain = pso_prey.swarm_best_position

        list_of_pso_prey.append(pso_prey)

        print('Training predators, alternation: ', i+1, ' out of ', nbr_of_training_alternations)
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
            pso_pred.particle_best_value = pso_pred.run_all_particles("pred")
            pso_pred.swarm_best_position = pso_pred.positions_matrix[np.argmax(pso_pred.particle_best_value), :]

            pso_pred.positions_matrix = list_of_pso_pred[-1].positions_matrix
            pso_pred.velocity_matrix = list_of_pso_pred[-1].velocity_matrix
            pso_pred.inertia_weight = list_of_pso_pred[-1].inertia_weight


        pso_pred.run_pso("pred")
        best_pred_brain = pso_pred.swarm_best_position


        list_of_pso_pred.append(pso_pred)

        pso_data = {'list_of_pso_pred': list_of_pso_pred, 'list_of_pso_prey': list_of_pso_prey}
        with open('TrainingData.p', 'wb') as f:
            pickle.dump(pso_data, f)
            print('data saved!')


except KeyboardInterrupt:
    print('Training aborted for some reason')
except RuntimeError:
    traceback.print_exc()
finally:
    pso_data = {'list_of_pso_pred': list_of_pso_pred, 'list_of_pso_prey': list_of_pso_prey}
    with open('TrainingData.p', 'wb') as f:
        pickle.dump(pso_data, f)
        print('data saved!')