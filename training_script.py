import pickle
from PSO_class import PSO
from Aquarium import aquarium


aquarium_parameters = {'nbr_of_prey': 15, 'nbr_of_pred': 2, 'size_X': 5, 'size_Y': 5, 'max_speed_prey': 0.05,
                       'max_speed_pred': 0.1, 'max_acc_prey': 0.2, 'max_acc_pred': 0.1, 'eat_radius': 0.05,
                       'weight_range': 5, 'nbr_of_hidden_neurons': 5, 'nbr_of_outputs': 2,
                       'visibility_range': 0.5, 'rand_walk_brain_set': [],
                       'input_set': ["enemy_pos", "friend_pos", "wall"]}

list_of_pso_prey = list()
list_of_pso_pred = list()
nbr_of_training_alternations = 2


try:
    for i in range(nbr_of_training_alternations):
        print('Training preys, iteration: ', i+1, ' out of ', nbr_of_training_alternations)
        # Train prey
        if i != 0:
            aquarium_parameters['rand_walk_brain_set'] = []
            pso_prey = PSO(aquarium_parameters=aquarium_parameters, train_prey=True)
            for i_aquarium in pso_prey.list_of_aquarium:
                i_aquarium.pred_brain.update_brain(best_pred_brain)
            for i_aquarium in pso_prey.list_of_validation_aquarium:
                i_aquarium.pred_brain.update_brain(best_pred_brain)
        else:
            aquarium_parameters['rand_walk_brain_set'] = ['pred']
            pso_prey = PSO(aquarium_parameters=aquarium_parameters, train_prey=True)

        pso_prey.run_pso()
        best_prey_brain = pso_prey.get_particle_position_with_best_val_fitness()

        list_of_pso_prey.append(pso_prey)

        print('Training predators, iteration: ', i+1, ' out of ', nbr_of_training_alternations)
        # Train predator
        pso_pred = PSO(aquarium_parameters=aquarium_parameters, train_prey=False)
        for i_aquarium in pso_pred.list_of_aquarium:
            i_aquarium.prey_brain.update_brain(best_prey_brain)
        for i_aquarium in pso_pred.list_of_validation_aquarium:
            i_aquarium.prey_brain.update_brain(best_prey_brain)
        pso_pred.run_pso()
        best_pred_brain = pso_pred.get_particle_position_with_best_val_fitness()

        list_of_pso_pred.append(pso_pred)
except KeyboardInterrupt:
    print('Training aborted')

pso_data = {'list_of_pso_prey': list_of_pso_prey,
            'list_of_pso_pred': list_of_pso_pred}

with open('TrainingData.p', 'wb') as f:
    pickle.dump(pso_data, f)

