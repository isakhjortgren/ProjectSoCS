import pickle
import numpy as np
import matplotlib.pyplot as plt


with open('FishDataFromPreviousFilm.p', 'rb') as f:
    fish_data = pickle.load(f)


pos_over_time = fish_data['pos_over_time']
vel_over_time = fish_data['vel_over_time']
nbr_prey = fish_data['nbr_prey']
nbr_pred = fish_data['nbr_pred']


def calculate_variance_of_prey():
    length = len(pos_over_time)
    array_of_var = np.zeros(length)
    array_of_time = np.linspace(0, length/50, length)
    for i in range(length):
        prey_pos = pos_over_time[i][nbr_pred:, :]
        mean_pos = np.mean(prey_pos, axis=0)

        pos_relative_to_center = prey_pos-mean_pos
        var_i = np.var(np.linalg.norm(pos_relative_to_center, axis=1))
        array_of_var[i] = var_i

    plt.plot(array_of_time, array_of_var)
    plt.axis([0, np.max(array_of_time), 0, 0.7])
    plt.title('Radial variance of prey position')
    plt.xlabel('Time')
    plt.ylabel('Variance')
    plt.show()




if __name__ == '__main__':
    calculate_variance_of_prey()