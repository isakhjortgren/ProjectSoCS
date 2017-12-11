import pickle
import numpy as np
from numpy import matlib
import matplotlib.pyplot as plt
import itertools

with open('respawn_data.p', 'rb') as f:
    fish_data = pickle.load(f)

pos_over_time = fish_data['pos_over_time']
vel_over_time = fish_data['vel_over_time']
nbr_prey = fish_data['nbr_prey']
nbr_pred = fish_data['nbr_pred']
score = fish_data["score"]
fish_eaten =  fish_data["fishes_eaten"]
time_array = fish_data['time']
figure_index = 1



def calculate_dilation_of_prey():
    test_3dMat = pos_over_time[:,nbr_pred:, :]
    positions_adjusted = np.copy(test_3dMat)
    mean_pos = test_3dMat.mean(axis=1)
    for i in range(test_3dMat.shape[1]):
        positions_adjusted[:, i, :] -= mean_pos

    radial_from_center = np.linalg.norm(positions_adjusted, axis=2)
    radial_mean = radial_from_center.mean(axis=1)
    radial_var = np.var(radial_from_center, axis=1)

    plt.figure()
    plt.plot(time_array, radial_mean)
    plt.fill_between(time_array, radial_mean - radial_var, radial_mean + radial_var, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.title('Radial dilation of prey position')
    plt.xlabel('Time')
    plt.ylabel('Dilation')
    print('jao')
    

def calc_corr():
    N = pos_over_time.shape[1]
    T = pos_over_time.shape[0]
    prey_correlation_x = np.corrcoef(pos_over_time[:, nbr_pred:, 0],rowvar=False)
    prey_correlation_y = np.corrcoef(pos_over_time[:, nbr_pred:, 1],rowvar=False)
    pred_correlation_x = np.corrcoef(pos_over_time[:, :nbr_pred, 0],rowvar=False)
    pred_correlation_y = np.corrcoef(pos_over_time[:, :nbr_pred, 1],rowvar=False)

    #Set diagonal to NaN
    i_es = list(range(nbr_prey))
    prey_correlation_x[i_es,i_es] = 100
    prey_correlation_y[i_es,i_es] = 100
    i_es = list(range(nbr_pred))
    pred_correlation_x[i_es,i_es] = 100
    pred_correlation_y[i_es,i_es] = 100


    resolution = 30

    plt.figure()
    plt.subplot(221)
    plt.hist(prey_correlation_x.reshape(prey_correlation_x.size),resolution, range=(-1,1))
    plt.title("Prey X Correlation")
    plt.subplot(222)
    plt.hist(prey_correlation_y.reshape(prey_correlation_y.size),resolution, range=(-1,1))
    plt.title("Prey Y Correlation")

    plt.subplot(223)
    plt.hist(pred_correlation_x.reshape(pred_correlation_x.size),resolution, range=(-1,1))
    plt.title("Pred X Correlation")
    plt.subplot(224)
    plt.hist(pred_correlation_y.reshape(pred_correlation_y.size),resolution, range=(-1,1))
    plt.title("Pred Y Correlation")

def histogram_of_positions():
    plt.figure()
    test_3dMat = pos_over_time[:,nbr_pred: , :]
    all_x_pos = test_3dMat[:, :, 0]
    all_x_pos = all_x_pos.reshape(all_x_pos.size)
    all_y_pos = test_3dMat[:, :, 1]
    all_y_pos = all_y_pos.reshape(all_y_pos.size)
    plt.figure()
    plt.hist2d(all_x_pos, all_y_pos, bins=160)
    plt.title('Position distribution')
    plt.colorbar()


pos_over_time_prey = pos_over_time[:,nbr_pred:,:]
pos_over_time_pred = pos_over_time[:,0:nbr_pred,:]
vel_over_time_prey = vel_over_time[:, nbr_pred:, :]
vel_over_time_pred = vel_over_time[:, 0:nbr_pred, :]

mean_pos_over_time_prey = np.mean(pos_over_time_prey, axis=1)
mean_pos_over_time_pred = np.mean(pos_over_time_pred, axis=1)
mean_vel_over_time_prey = np.mean(vel_over_time_prey, axis=1)
mean_vel_over_time_pred = np.mean(vel_over_time_pred, axis=1)


def calculate_rotaion_and_polarization():
    number_of_timesteps = pos_over_time_prey.shape[0]
    number_of_prey = pos_over_time_prey.shape[1]

    normalised_vel_over_time_prey = vel_over_time_prey / np.linalg.norm(vel_over_time_prey, axis=2)[:, :, np.newaxis]
    normalised_mean_vel_over_time_prey = np.mean(normalised_vel_over_time_prey, axis=1)

    polarisation_over_time_prey = np.linalg.norm(normalised_mean_vel_over_time_prey, axis=1)

    pos_relative_mean_over_time_prey = pos_over_time_prey - mean_pos_over_time_prey[:, np.newaxis, :]
    pos_relative_mean_over_time_prey = pos_relative_mean_over_time_prey/np.linalg.norm(pos_relative_mean_over_time_prey,axis=2)[:,:,np.newaxis]

    z = np.zeros(shape=(number_of_timesteps, number_of_prey))
    for t in range(number_of_timesteps):
        for prey in range(number_of_prey):
            x1 = normalised_vel_over_time_prey[t,prey,0]
            x2 = pos_relative_mean_over_time_prey[t,prey,0]
            y1 = normalised_vel_over_time_prey[t,prey,1]
            y2 = pos_relative_mean_over_time_prey[t, prey, 1]
            z[t,prey] = x1*y2-x2*y1
    rotation_over_time_prey = np.mean(z,1)

    plt.figure()
    plt.subplot(121)
    plt.plot(time_array, polarisation_over_time_prey)
    plt.title("Prey Polarisation")

    plt.subplot(122)
    plt.plot(time_array, rotation_over_time_prey)
    plt.title("Prey Rotation")
    plt.show()





if __name__ == '__main__':
    calculate_rotaion_and_polarization()
    #histogram_of_positions()
    #calc_corr()
    #calculate_dilation_of_prey()
    #plt.show()
