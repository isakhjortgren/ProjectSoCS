import pickle
import numpy as np
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
        self.fish_eaten =  fish_data["fishes_eaten"]
        self.time_array = fish_data['time']
    
    
    
    def calculate_dilation_of_prey(self):
        test_3dMat = self.pos_over_time[:,self.nbr_pred:, :]
        positions_adjusted = np.copy(test_3dMat)
        mean_pos = test_3dMat.mean(axis=1)
        for i in range(test_3dMat.shape[1]):
            positions_adjusted[:, i, :] -= mean_pos
    
        radial_from_center = np.linalg.norm(positions_adjusted, axis=2)
        radial_mean = radial_from_center.mean(axis=1)
        radial_var = np.std(radial_from_center, axis=1)
    
        plt.figure()
        plt.plot(self.time_array, radial_mean)
        plt.fill_between(self.time_array, radial_mean - radial_var, radial_mean + radial_var, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
        plt.title('Radial dilation of prey position')
        plt.xlabel('Time')
        plt.ylabel('Dilation')
        print('jao')
        
    
    def calc_corr(self):
        N = self.pos_over_time.shape[1]
        T = self.pos_over_time.shape[0]
        prey_correlation_x = np.corrcoef(self.pos_over_time[:, self.nbr_pred:, 0],rowvar=False)
        prey_correlation_y = np.corrcoef(self.pos_over_time[:, self.nbr_pred:, 1],rowvar=False)
        pred_correlation_x = np.corrcoef(self.pos_over_time[:, :self.nbr_pred, 0],rowvar=False)
        pred_correlation_y = np.corrcoef(self.pos_over_time[:, :self.nbr_pred, 1],rowvar=False)
    
        #Set diagonal to NaN
        i_es = list(range(self.nbr_prey))
        prey_correlation_x[i_es,i_es] = 100
        prey_correlation_y[i_es,i_es] = 100
        i_es = list(range(self.nbr_pred))
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
    
    def histogram_of_positions(self):
        test_3dMat = self.pos_over_time[:,self.nbr_pred: , :]
        all_x_pos = test_3dMat[:, :, 0]
        all_x_pos = all_x_pos.reshape(all_x_pos.size)
        all_y_pos = test_3dMat[:, :, 1]
        all_y_pos = all_y_pos.reshape(all_y_pos.size)
        plt.figure()
        plt.hist2d(all_x_pos, all_y_pos, bins=160)
        plt.title('Position distribution')
        plt.colorbar()


if __name__ == '__main__':
    a = AnalyzeClass('respawn_data.p')
    a.histogram_of_positions()
    a.calc_corr()
    a.calculate_dilation_of_prey()
    plt.show()
