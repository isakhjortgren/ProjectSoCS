import pickle
from Aquarium import aquarium
import matplotlib.pyplot as plt
import matplotlib

with open('TrainingData.p', 'rb') as f:
    pso_data = pickle.load(f)

list_of_pso_pred = pso_data['list_of_pso_pred']
list_of_pso_prey = pso_data['list_of_pso_prey']

def visulaize_all_training_aquarium():

    fig, axes = plt.subplots(nrows=max(len(list_of_pso_prey), 2), ncols=2)

    for i_pso_prey, pso_prey in enumerate(list_of_pso_prey):
        axes[i_pso_prey, 0].plot(pso_prey.list_of_swarm_best_value, 'b', label='Training value')
        axes[i_pso_prey, 0].plot(pso_prey.list_of_validation_results, 'r', label='Validation value')

    axes[0, 0].set_title('Fitness value for preys')

    for i_pso_pred, pso_pred in enumerate(list_of_pso_pred):
        axes[i_pso_pred, 1].plot(pso_pred.list_of_swarm_best_value, 'b', label='Training value')
        axes[i_pso_pred, 1].plot(pso_pred.list_of_validation_results, 'r', label='Validation value')

    axes[0, 1].set_title('Fitness value for predators')

    plt.savefig('FitnessPlots.png')
    """
    for i in range(len(pso_prey.list_of_aquarium)):
        aquarium_1 = pso_prey.list_of_aquarium[i]
        best_prey_brain = pso_prey.get_particle_position_with_best_val_fitness()
        aquarium_1.prey_brain.update_brain(best_prey_brain)
        aquarium_1.set_videoutput('last_trained_prey_aq%s.mp4'%i)
        print(aquarium_1.run_simulation())
        aquarium_1 = pso_pred.list_of_aquarium[i]
        best_pred_brain = pso_pred.get_particle_position_with_best_val_fitness()
        aquarium_1.pred_brain.update_brain(best_pred_brain)
        aquarium_1.set_videoutput('last_trained_pred_aq%s.mp4'%i)

        print(aquarium_1.run_simulation())
    """

def visulaize_one_aquarium(val_aquarium=0):
    pso_prey = list_of_pso_prey[-1]
    pso_pred = list_of_pso_pred[-1]
    aquarium_1 = pso_prey.list_of_validation_aquarium[val_aquarium]
    #best_pred_brain = pso_pred.get_particle_position_with_best_val_fitness()
    best_pred_brain = pso_pred.swarm_best_position
    aquarium_1.pred_brain.update_brain(best_pred_brain)

    #best_prey_brain = pso_prey.get_particle_position_with_best_val_fitness()
    best_prey_brain = pso_prey.swarm_best_position
    aquarium_1.prey_brain.update_brain(best_prey_brain)
    aquarium_1.set_videoutput('validation_aquarium_nr_%s.mp4' % val_aquarium)
    aquarium_1.run_simulation()

def visulaize_a_new_aquarium():
    pso_prey = list_of_pso_prey[-1]
    pso_pred = list_of_pso_pred[-1]
    aquarium_1 = aquarium(**pso_pred.aquarium_parameters)
    print(pso_pred.aquarium_parameters['input_set'])
    print('safe boundary? ', pso_pred.aquarium_parameters['safe_boundary'])
    # best_pred_brain = pso_pred.get_particle_position_with_best_val_fitness()
    best_pred_brain = pso_pred.get_particle_position_with_best_val_fitness()
    aquarium_1.pred_brain.update_brain(best_pred_brain)

    # best_prey_brain = pso_prey.get_particle_position_with_best_val_fitness()
    best_prey_brain = pso_prey.swarm_best_position
    aquarium_1.prey_brain.update_brain(best_prey_brain)
    aquarium_1.set_videoutput('new_aquarium.mp4', fps=45)

    aquarium_1.run_simulation()


if __name__ == '__main__':
    visulaize_a_new_aquarium()
