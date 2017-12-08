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
    plt.title('Radial variance of prey position')
    plt.xlabel('Time')
    plt.ylabel('Variance')
    

def calc_corr():
    N=20

    x_pos = np.array(size=(pos_over_time[0].shape[0],len(pos_over_time)))
    correlation_x = np.x_pos



    for i,j in itertools.product(range(N), range(N)):
        correlation_x = np.correlate(pos_over_time[:],i,],pos_over_time[j,0,:]   )     

    plt.hist(correlation_x,10)

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

if __name__ == '__main__':
    #calculate_variance_of_prey()
    
    x_pos, y_pos = hotfix(pos_over_time)
    print()
    exit()
    calc_corr()
    
    plt.show()
