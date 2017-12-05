import pickle
from PSO_class import PSO
import traceback
from Aquarium import aquarium


aquarium_parameters = {'nbr_of_prey': 15, 'nbr_of_pred': 2, 'size_X': 2, 'size_Y': 2, 'max_speed_prey': 0.15,
                       'max_speed_pred': 0.2, 'max_acc_prey': 0.3, 'max_acc_pred': 0.5, 'eat_radius': 0.05,
                       'weight_range': 1, 'nbr_of_hidden_neurons': 5, 'nbr_of_outputs': 2,
                       'visibility_range': 0.5, 'rand_walk_brain_set': ['prey'],
                       'input_set': ["enemy_pos", "friend_pos", "wall", "enemy_vel"], 'safe_boundary': True}


list_of_pso_pred = list()
nbr_of_training_alternations = 2


try:
    for i in range(nbr_of_training_alternations):

        print('Training predators against dodgebrain, iteration: ', i+1, ' out of ', nbr_of_training_alternations)
        # Train predator
        pso_pred = PSO(aquarium_parameters=aquarium_parameters, train_prey=False)
        pso_pred.run_pso()

        list_of_pso_pred.append(pso_pred)
except KeyboardInterrupt:
    print('Training aborted for some reason')
finally:
    pso_data = {'list_of_pso_pred': list_of_pso_pred}
    with open('TrainingData.p', 'wb') as f:
        pickle.dump(pso_data, f)
        print('data saved!')

