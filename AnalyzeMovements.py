import pickle
import numpy as np
import matplotlib.pyplot as plt
import itertools

with open('FishDataFromPreviousFilm.p', 'rb') as f:
    fish_data = pickle.load(f)


pos_over_time = fish_data['pos_over_time']
vel_over_time = fish_data['vel_over_time']
nbr_prey = fish_data['nbr_prey']
nbr_pred = fish_data['nbr_pred']
fishes_removed =  list(reversed(fish_data["fishes_removed"]))


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
    plt.title('Radial dilation of prey position')
    plt.xlabel('Time')
    plt.ylabel('Dilation')

def calculate_dilation_of_prey():
    test_3dMat = np.random.rand(10000, 10, 2)
    array_of_time = np.linspace(0, 1, 10000)
    positions_adjusted = np.copy(test_3dMat)
    mean_pos = test_3dMat.mean(axis=1)
    for i in range(test_3dMat.shape[1]):
        positions_adjusted[:, i, :] -= mean_pos

    radial_from_center = np.linalg.norm(positions_adjusted, axis=2)
    radial_mean = radial_from_center.mean(axis=1)
    radial_var = np.var(radial_from_center, axis=1)
    plt.figure(1)
    plt.plot(array_of_time, radial_mean)
    plt.fill_between(array_of_time, radial_mean - radial_var, radial_mean + radial_var, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.title('Radial dilation of prey position')
    plt.xlabel('Time')
    plt.ylabel('Dilation')
    print('jao')
    

def calc_corr(x_pos, y_pos):
    N = x_pos.shape[0]
    T = x_pos.shape[1]
    prey_correlation_x = np.corrcoef(x_pos[3:,T//2:])
    prey_correlation_y = np.corrcoef(y_pos[3:,T//2:])
    pred_correlation_x = np.corrcoef(x_pos[:3,T//2:])
    pred_correlation_y = np.corrcoef(y_pos[:3,T//2:])
    


    #Set diagonal to NaN
    i_es = list(range(N-3))
    prey_correlation_x[i_es,i_es] = 100
    prey_correlation_y[i_es,i_es] = 100
    i_es = list(range(3))
    pred_correlation_x[i_es,i_es] = 100
    pred_correlation_y[i_es,i_es] = 100

    
    plt.subplot(221)
    plt.hist(prey_correlation_x.reshape((N-3)**2),30, range=(-1,1))
    plt.title("Prey X Correlation")
    plt.subplot(222)
    plt.hist(prey_correlation_y.reshape((N-3)**2),30, range=(-1,1))
    plt.title("Prey Y Correlation")

    plt.subplot(223)
    plt.hist(pred_correlation_x.reshape(9),30, range=(-1,1))
    plt.title("Pred X Correlation")
    plt.subplot(224)
    plt.hist(pred_correlation_y.reshape(9),30, range=(-1,1))
    plt.title("Pred Y Correlation")

    plt.show()

def hotfix(t_pos):
    T= len(t_pos)
    fishes = t_pos[0].shape[0]

    x_pos = np.zeros( (fishes, T) )
    y_pos = np.zeros( (fishes, T) )

    for i, fish_xy in enumerate( t_pos):
        if fish_xy.shape[0] != x_pos.shape[0]:
            ### HOTFIX TIME
            indicies = fishes_removed.pop()
            x_pos = np.delete(x_pos, indicies, axis = 0)
            y_pos = np.delete(y_pos, indicies, axis = 0)
        x_pos[:,i] = fish_xy[:,0]
        y_pos[:,i] = fish_xy[:,1]  
    return x_pos,y_pos


def histogram_of_positions():
    test_3dMat = np.random.rand(10000, 10, 2)
    all_x_pos = test_3dMat[:, :, 0]
    all_x_pos = all_x_pos.reshape(all_x_pos.size)
    all_y_pos = test_3dMat[:, :, 1]
    all_y_pos = all_y_pos.reshape(all_y_pos.size)
    print('hej')
    plt.hist2d(all_x_pos, all_y_pos, bins=40)
    plt.title('Position distribution')
    plt.colorbar()


if __name__ == '__main__':
    calculate_dilation_of_prey()
    plt.show()
