import pickle
from Aquarium import aquarium
import matplotlib.pyplot as plt
import matplotlib

with open('TrainingData.p', 'rb') as f:
    pso_data = pickle.load(f)

list_of_pso_prey = pso_data['list_of_pso_prey']

fig, axes = plt.subplots(nrows=len(list_of_pso_prey), ncols=2)

for i_pso_prey, pso_prey in enumerate(list_of_pso_prey):
    axes[i_pso_prey, 0].plot(pso_prey.list_of_swarm_best_value, 'b.', label='Training value')
    #axes[i_pso_prey, 0].plot(pso_prey.list_of_validation_results, 'r.', label='Validation value')
axes[0, 0].set_title('Fitness value for preys')

list_of_pso_pred = pso_data['list_of_pso_pred']
for i_pso_pred, pso_pred in enumerate(list_of_pso_pred):
    axes[i_pso_pred, 1].plot(pso_pred.list_of_swarm_best_value, 'b.', label='Training value')
    #axes[i_pso_pred, 1].plot(pso_pred.list_of_validation_results, 'r.', label='Validation value')
axes[0, 1].set_title('Fitness value for predators')

plt.show()

aquarium_1 = pso_prey.list_of_aquarium[0]
aquarium_1.pred_brain.update_brain(pso_prey.swarm_best_position)
aquarium_1.set_videoutput('last_trained_prey.mp4')
print(aquarium_1.run_simulation())
aquarium_1 = pso_pred.list_of_aquarium[0]
aquarium_1.pred_brain.update_brain(pso_pred.swarm_best_position)
aquarium_1.set_videoutput('last_trained_pred.mp4')

print(aquarium_1.run_simulation())
