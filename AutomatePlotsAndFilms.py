import pickle
from RespawnAquarium import respawnAquarium




def generate_movement_data_and_video(list_of_data_sets):
    for i_data_set in list_of_data_sets:
        print('generate video and movement data for data set %s' % i_data_set)
        training_data_file_name = 'TrainingData%s.p' % i_data_set
        with open('TrainingData.p', 'rb') as f:
            pso_data = pickle.load(f)

        list_of_pso_pred = pso_data['list_of_pso_pred']
        list_of_pso_prey = pso_data['list_of_pso_prey']
        pso_prey = list_of_pso_prey[-1]
        pso_pred = list_of_pso_pred[-1]

        aq_par = pso_prey.aquarium_parameters

        aq_par["outfile"] = "MovementData%s.p" % i_data_set
        aq_par["sim_time"] = 15

        size = 2
        aq_par["size_X"] = size
        aq_par["size_Y"] = size

        aq = respawnAquarium(**aq_par)
        aq.pred_brain.update_brain(pso_pred.get_particle_position_with_best_val_fitness())
        aq.prey_brain.update_brain(pso_prey.get_particle_position_with_best_val_fitness())

        # aq.run_simulation()
        aq.run_simulation_video("video_%s.mp4" % i_data_set)

if __name__ == '__main__':
    list_of_data_sets = [2]  # 1, 2, 3, 7, 8, 9, 10, 11]
    generate_movement_data_and_video(list_of_data_sets)