from Aquarium import aquarium
import numpy as np

aquarium_paramters = {'nbr_of_prey': 3, 'nbr_of_pred': 2, 'size_X': 1, 'size_Y': 1, 'max_speed_prey': 0.01,
                      'max_speed_pred': 0.1, 'max_acc_prey': 0.1, 'max_acc_pred': 0.1, 'eat_radius': 0.1,
                      'weight_range': 5, 'nbr_of_hidden_neurons': 10, 'nbr_of_inputs': 10, 'nbr_of_outputs': 2}

a = aquarium(**aquarium_paramters)
a.fish_xy = np.copy(a.fish_xy_start)
a.fish_vel = np.zeros(a.fish_xy_start.shape)

a.interval_pred = list(range(a.nbr_of_pred))
a.interval_prey = list(range(a.nbr_of_pred,a.nbr_of_prey+a.nbr_of_pred))
print(a.calculate_inputs())
print('hej')