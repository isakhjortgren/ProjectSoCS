import pickle
import numpy as np
from numpy import matlib
import matplotlib.pyplot as plt
import itertools

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
        self.time_array = fish_data['time']


        try:
            self.size = fish_data["size"]
        except:
            self.size = round(np.max(self.pos_over_time[:,:,:]))


        self.figure_name_beginning = data_file.replace('.p', '')



    def calculate_sub_graphs(self):
        fish_pos_t = self.pos_over_time[:, self.nbr_pred:, :]
        threshold = 0.3

        def find_clusters(L):
            not_connected_to_0 = L[:,0] == 0
            send_to_next = L[not_connected_to_0, :][:, not_connected_to_0]
            if send_to_next.size == 0:
                return 1
            return find_clusters(send_to_next) + 1

        nbr_cluster_size = np.zeros(fish_pos_t.shape[0])
        for i, fish_xy in enumerate(fish_pos_t):
            x_diff = np.column_stack([fish_xy[:, 0]] * self.nbr_prey) - np.row_stack([fish_xy[:, 0]] * self.nbr_prey)
            y_diff = np.column_stack([fish_xy[:, 1]] * self.nbr_prey) - np.row_stack([fish_xy[:, 1]] * self.nbr_prey)
            distances = np.sqrt(x_diff ** 2 + y_diff ** 2)
            A = distances < threshold
            A_n = np.linalg.matrix_power(A, A.shape[0])

            Lij = (A_n > 0.5).astype(int)
            nbr_cluster_size[i] = find_clusters(Lij)
        plt.figure()
        plt.plot(self.time_array, nbr_cluster_size)
        plt.ylim(0, self.nbr_prey)
        plt.savefig(self.figure_name_beginning + 'subGraph.png')







    def calculate_dilation_of_prey(self):
        test_3dMat = self.pos_over_time[:,self.nbr_pred:, :]
        positions_adjusted = np.copy(test_3dMat)
        mean_pos = test_3dMat.mean(axis=1)
        for i in range(test_3dMat.shape[1]):
            positions_adjusted[:, i, :] -= mean_pos

        radial_from_center = np.linalg.norm(positions_adjusted, axis=2)
        radial_mean = radial_from_center.mean(axis=1)
        radial_max = np.max(radial_from_center, axis=1)
        radial_min = np.min(radial_from_center, axis=1)
    
        plt.figure(dpi=180)
        plt.plot(self.time_array, radial_mean)
        plt.fill_between(self.time_array, radial_min, radial_max, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
        plt.title('Radial dilation of prey position')
        plt.xlabel('Time')
        plt.ylabel('Dilation')
        print('jao')
        plt.tight_layout()
        plt.savefig(self.figure_name_beginning + "_calclte_Dajläjtion.png")
    
    def calc_corr(self):
        N = self.pos_over_time.shape[1]
        T = self.pos_over_time.shape[0]
        prey_correlation_x = np.corrcoef(self.pos_over_time[:, self.nbr_pred:, 0],rowvar=False)
        prey_correlation_y = np.corrcoef(self.pos_over_time[:, self.nbr_pred:, 1],rowvar=False)
        
        """
        pred_correlation_x = np.corrcoef(self.pos_over_time[:, :self.nbr_pred, 0],rowvar=False)
        pred_correlation_y = np.corrcoef(self.pos_over_time[:, :self.nbr_pred, 1],rowvar=False)
        """


        #Set diagonal to NaN
        i_es = list(range(self.nbr_prey))
        prey_correlation_x[i_es,i_es] = 100
        prey_correlation_y[i_es,i_es] = 100
        
        """
        i_es = list(range(self.nbr_pred))
        pred_correlation_x[i_es,i_es] = 100
        pred_correlation_y[i_es,i_es] = 100
        """
    
        resolution = 30
    
        plt.figure(num=None, figsize=(4, 7), dpi=180, facecolor='w', edgecolor='k')
        
        plt.subplot(211)
        freq = plt.hist(prey_correlation_x.reshape(prey_correlation_x.size),resolution, range=(-1,1), weights=100*np.ones(prey_correlation_x.size)/prey_correlation_x.size)
        plt.text(-1, 0.9*np.max(freq[0]) ,"Prey y-coordinate\ncorrelation histogram")
        #plt.xlabel("s")
        plt.ylabel("Frequency [%]")

        plt.subplot(212)
        freq =plt.hist(prey_correlation_y.reshape(prey_correlation_y.size),resolution, range=(-1,1), weights=100*np.ones(prey_correlation_y.size)/prey_correlation_y.size)
        plt.text(-1, 0.9*np.max(freq[0]) ,"Prey y-coordinate\ncorrelation histogram")
        plt.ylabel("Frequency [%]")
    
        """
        plt.subplot(223)
        plt.hist(pred_correlation_x.reshape(pred_correlation_x.size),resolution, range=(-1,1))
        plt.title("Pred X Correlation")
        plt.subplot(224)
        plt.hist(pred_correlation_y.reshape(pred_correlation_y.size),resolution, range=(-1,1))
        plt.title("Pred Y Correlation")
        """ 
        plt.tight_layout()
        
        plt.savefig(self.figure_name_beginning + "_posCorr.png")

    
    def histogram_of_positions(self):
        test_3dMat = self.pos_over_time[:,self.nbr_pred: , :]
        all_x_pos = test_3dMat[:, :, 0]
        all_x_pos = all_x_pos.reshape(all_x_pos.size)
        all_y_pos = test_3dMat[:, :, 1]
        all_y_pos = all_y_pos.reshape(all_y_pos.size)
        fig = plt.figure(dpi=180)
        plt.hist2d(all_x_pos, all_y_pos, bins=160, range=[(0, self.size), (0,self.size)])

        plt.axis('off')
        plt.title('Position distribution')
        #plt.colorbar()
        plt.tight_layout()
        plt.savefig(self.figure_name_beginning + "_posHisto.png")



    def calculate_rotation_and_polarization(self):
        pos_over_time_prey = self.pos_over_time[:, self.nbr_pred:, :]
        pos_over_time_pred = self.pos_over_time[:, 0:self.nbr_pred, :]
        vel_over_time_prey = self.vel_over_time[:, self.nbr_pred:, :]
        vel_over_time_pred = self.vel_over_time[:, 0:self.nbr_pred, :]

        mean_pos_over_time_prey = np.mean(pos_over_time_prey, axis=1)
        #mean_pos_over_time_pred = np.mean(pos_over_time_pred, axis=1)
        #mean_vel_over_time_prey = np.mean(vel_over_time_prey, axis=1)
        #mean_vel_over_time_pred = np.mean(vel_over_time_pred, axis=1)

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

        plt.figure(dpi=180)
        plt.subplot(121)
        plt.plot(self.time_array, polarisation_over_time_prey)
        plt.title("Prey Polarisation")

        plt.subplot(122)
        plt.plot(self.time_array, rotation_over_time_prey)
        plt.title("Prey Rotation")


if __name__ == '__main__':
    a = AnalyzeClass('MovementData2.p')
    #a.calculate_rotation_and_polarization()
    #a.histogram_of_positions()
    #a.calc_corr()
    a.calculate_sub_graphs()



