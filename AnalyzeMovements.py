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
        self.fish_eaten = fish_data["fishes_eaten"]
        self.time_array = fish_data['time']


        try:
            self.size = fish_data["size"]
        except:
            self.size = round(np.max(self.pos_over_time[:,:,:]))


        self.figure_name_beginning = data_file.replace('.p', '')

    
    
    
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
        plt.show()
        #plt.savefig(self.figure_name_beginning + "_posCorr.png")

    
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
        plt.savefig(self.figure_name_beginning + "_posHisto.png")

if __name__ == '__main__':

    a = AnalyzeClass('respawn_data.p')
    a.histogram_of_positions()
    a.calc_corr()

    a.calculate_dilation_of_prey()
    plt.show()
