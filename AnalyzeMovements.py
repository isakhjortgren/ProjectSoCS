import pickle
import numpy as np
from numpy import matlib
import matplotlib.pyplot as plt
import itertools
from matplotlib.gridspec import GridSpec


try:
    with open("cluster_sizes.p","rb") as f:
        cluster_size_dict = pickle.load(f)

except:
    cluster_size_dict = {}

class AnalyzeClass(object):
    def __init__(self, data_file):
        with open(data_file, 'rb') as f:
            fish_data = pickle.load(f)
        
        self.pos_over_time = fish_data['pos_over_time']
        self.vel_over_time = fish_data['vel_over_time']
        self.nbr_prey = fish_data['nbr_prey']
        self.nbr_pred = fish_data['nbr_pred']
        self.score = fish_data["score"]
        self.fish_eaten = fish_data["fishes_eaten"]
        try:
            self.time_array = fish_data['time']
        except: 
            self.time_array = np.linspace(0, 1000, self.pos_over_time.shape[0])

        try:
            self.size = fish_data["size"]
        except:
            self.size = round(np.max(self.pos_over_time[:,:,:]))


        self.figure_name_beginning = data_file.replace('.p', '')

    def calculate_sub_graphs(self):
        fish_pos_t = self.pos_over_time[:, self.nbr_pred:, :]
        threshold = 0.3


        if self.figure_name_beginning in cluster_size_dict: 
            largest_cluster_size = cluster_size_dict[self.figure_name_beginning]
        else:
            largest_cluster_size = np.zeros(fish_pos_t.shape[0])
            for i, fish_xy in enumerate(fish_pos_t):
                x_diff = np.column_stack([fish_xy[:, 0]] * self.nbr_prey) - np.row_stack([fish_xy[:, 0]] * self.nbr_prey)
                y_diff = np.column_stack([fish_xy[:, 1]] * self.nbr_prey) - np.row_stack([fish_xy[:, 1]] * self.nbr_prey)
                distances = np.sqrt(x_diff ** 2 + y_diff ** 2)
                A = distances < threshold
                A_n = np.linalg.matrix_power(A, A.shape[0])

                Lij = (A_n > 0.5).astype(int)
                largest_cluster_size[i] = np.max(Lij.sum(axis=0))

        cluster_size_dict[self.figure_name_beginning] = np.copy(largest_cluster_size)
        bins = list(range(0,self.nbr_prey+1))
        
        #plt.hist(largest_cluster_size, resolution)
        ret = plt.hist(largest_cluster_size,bins, \
                weights=100*np.ones(largest_cluster_size.size)/largest_cluster_size.size)

        if np.max(ret[0])> 15:
            plt.ylim([0, 100])
        elif np.max(ret[0])> 5:
            plt.ylim([0, 15])
        else:
            plt.ylim([0, 5])


    def calculate_dilation_of_prey(self, ylim):
        test_3dMat = self.pos_over_time[:,self.nbr_pred:, :]
        positions_adjusted = np.copy(test_3dMat)
        mean_pos = test_3dMat.mean(axis=1)
        for i in range(test_3dMat.shape[1]):
            positions_adjusted[:, i, :] -= mean_pos

        radial_from_center = np.linalg.norm(positions_adjusted, axis=2)
        radial_mean = radial_from_center.mean(axis=1)
        radial_max = np.max(radial_from_center, axis=1)
        radial_min = np.min(radial_from_center, axis=1)

        plt.plot(self.time_array, radial_mean)
        plt.fill_between(self.time_array, radial_min, radial_max, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
        plt.title('Radial dilation of prey position')
        plt.xlabel('Time')
        plt.ylabel('Dilation')
        plt.ylim([0, ylim])
        
        for i in self.fish_eaten:
            t = self.time_array[i]
            plt.plot([t, t], [ylim, ylim*9/10],'r-')

        print('jao')

    def calc_corr(self, ylim):
        N = self.pos_over_time.shape[1]
        T = self.pos_over_time.shape[0]
        prey_correlation_x = np.corrcoef(self.pos_over_time[:, self.nbr_pred:, 0],rowvar=False)
        
        i_es = list(range(self.nbr_prey))
        prey_correlation_x[i_es,i_es] = 100
        
        resolution = 30
        freq = plt.hist(prey_correlation_x.reshape(prey_correlation_x.size),resolution, range=(-1,1), weights=100*np.ones(prey_correlation_x.size)/prey_correlation_x.size)
        #plt.text(-1, 0.9*np.max(freq[0]) ,"Prey x-coordinate\ncorrelation distribution")
        plt.ylim([0, ylim])
        
    
    def histogram_of_positions(self):
        test_3dMat = self.pos_over_time[:,self.nbr_pred: , :]
        all_x_pos = test_3dMat[:, :, 0]
        all_x_pos = all_x_pos.reshape(all_x_pos.size)
        all_y_pos = test_3dMat[:, :, 1]
        all_y_pos = all_y_pos.reshape(all_y_pos.size)
        
        plt.hist2d(all_x_pos, all_y_pos, bins=160, range=[(0, self.size), (0,self.size)])
        plt.axis('off')
        
    def calculate_rotation_and_polarization(self, iterations_to_plot, ylim, show_legend):
        start = self.pos_over_time.shape[0]//2
        end = start + iterations_to_plot

        pos_over_time_prey = self.pos_over_time[start:end, self.nbr_pred:, :]
        vel_over_time_prey = self.vel_over_time[start:end, self.nbr_pred:, :]
        
        time = self.time_array[start:end]

        mean_pos_over_time_prey = np.mean(pos_over_time_prey, axis=1)

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

       
        pol_line, = plt.plot(time, polarisation_over_time_prey, label="Prey Polarisation")
        rot_line, = plt.plot(time, abs(rotation_over_time_prey),label="Prey Rotation")
        plt.ylim([0, ylim])
        if show_legend:
            plt.legend(handles=[pol_line, rot_line])
 
if __name__ == '__main__sadf':
    
    dpi = 180
    configs = ["1","10","2", "7"]
    filenames = ["MovementData"+nbr+".p" for nbr in configs]

    save_fig = True

    plot_objects = [AnalyzeClass(filename) for filename in filenames]

    ## Histogram
    if 1 == 12:
        plt.figure(dpi=dpi)
        for i in range(4):
            plt.subplot(221+i)
            plot_objects[i].histogram_of_positions()
        plt.tight_layout()
        if save_fig:
            plt.savefig("Position_histogram.png")
        else:
            plt.show()

    #Correlation
    if 1 == 12:
        plt.figure(dpi=dpi)
        for i in range(4):
            plt.subplot(221+i)
            plot_objects[i].calc_corr(30)
            
            if i%2==0:
                plt.ylabel("Frequency [%]")
            else:
                frame1 = plt.gca()
                #frame1.axes.xaxis.set_ticklabels([])
                frame1.axes.yaxis.set_ticklabels([])        
        plt.tight_layout()
        if save_fig:
            plt.savefig("Correlation_x_pos.png")
        else:
            plt.show()

    
    #Rotation and polarization
    if 1 == 1:
        plt.figure(dpi=dpi)
        for i in range(4):
            plt.subplot(221+i)
            plot_objects[i].calculate_rotation_and_polarization(600, 1, i==0)   
        plt.tight_layout()
        if save_fig:
            plt.savefig("Pol_and_rot.png")
        else:
            plt.show()

    #Dilation
    if 1 == 12:
        plt.figure(dpi=dpi)
        for i in range(4):
            plt.subplot(221+i)
            plot_objects[i].calculate_dilation_of_prey(4)   
        plt.tight_layout()
        if save_fig:
            plt.savefig("Dilation.png")
        else:
            plt.show()

    # Graphs
    if 1 == 12:
        plt.figure(dpi=dpi)
        for i in range(4):
            plt.subplot(221+i)
            plot_objects[i].calculate_sub_graphs()   
        if save_fig:
            plt.savefig("Largest_cluster_histogram.png")
        else:
            plt.show()
        
        with open("cluster_sizes.p","wb") as f:
            pickle.dump(cluster_size_dict, f)
        
    print("done")


if __name__ == '__main__':
    dpi = 80
    filename = "MovementData7.p"
    plot_object = AnalyzeClass(filename)

    fish_txy = plot_object.pos_over_time

    import matplotlib.animation as animation  

    fig, ax = plt.subplots()
    scatter_plot_fish, = plt.plot(fish_txy[0,8:,0],fish_txy[0,8:,1], 'ob')
    scatter_plot_shark, = plt.plot(fish_txy[0,:8,0],fish_txy[0,:8,1], 'or')
    

    def animate(i):
        scatter_plot_fish.set_xdata(fish_txy[i,8:,0])  # update the data
        scatter_plot_fish.set_ydata(fish_txy[i,8:,1])  # update the data
        scatter_plot_shark.set_xdata(fish_txy[i,:8,0])  # update the data
        scatter_plot_shark.set_ydata(fish_txy[i,:8,1])  # update the data
        return scatter_plot_fish,scatter_plot_shark

    ani = animation.FuncAnimation(fig, animate, list(range(fish_txy.shape[0])),
                              interval=1, blit=True)
    plt.show()
