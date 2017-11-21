import numpy as np
import numpy.matlib
random = np.random.random

import matplotlib

import matplotlib.pyplot as plt
import matplotlib.animation as manimation

from Brain import Brain


class aquarium(object):
    # TODO:
    #  - kill simulations with useless sharks (For future!)
    #  - calculate dt with respect to eat_radius and max_speeds

    def __init__(self, nbr_of_prey, nbr_of_pred, size_X, size_Y,
                 max_speed_prey,max_speed_pred,max_acc_prey,max_acc_pred, 
                 eat_radius, nbr_of_hidden_neurons,nbr_of_inputs,nbr_of_outputs,
                 weight_range, visibility_range):


        #ToDo Init object variables
        self.record_video = False
        self.video_enabled = False
        self.size_X = size_X
        self.size_Y = size_Y

        self.pred_brain = Brain(nbr_of_hidden_neurons, nbr_of_inputs, nbr_of_outputs, weight_range) 
        self.prey_brain = Brain(nbr_of_hidden_neurons, nbr_of_inputs, nbr_of_outputs, weight_range) 

        self.eat_radius = eat_radius
        self.visibility_range = visibility_range

        self.nbr_of_prey = nbr_of_prey
        self.nbr_of_pred = nbr_of_pred

        self.max_vel_prey = max_speed_prey
        self.max_vel_pred = max_speed_pred
        
        self.max_acc_prey = max_acc_prey
        self.max_acc_pred = max_acc_pred

        #Constant
        self.fish_xy_start = np.matrix(random(size=(nbr_of_prey+nbr_of_pred,2)))\
                            *np.matrix([[size_X, 0], [0,size_Y]])
        
        self.brain_input = None  #TODO: Only for debug purposes 
        self.interval_pred = None
        self.interval_prey = None
        self.eaten = None 

        self.fish_xy = None
        self.fish_vel = None
        self.acc_fish = None

    def neighbourhood(self, distances):
        return np.exp(-distances**2/(2*self.visibility_range**2))/self.visibility_range

    def calculate_inputs(self):
        
        n_preds = self.interval_pred[-1] +1
        n_preys = len(self.interval_prey)

        N = len(self.fish_xy) # also equal to (n_preds + n_preys)

        return_matrix = np.zeros(( N, self.pred_brain.nbr_of_inputs))

        ## Differences ##
        x_diff = np.column_stack([self.fish_xy[:,0]]*N) - np.row_stack([self.fish_xy[:,0]]*N) 
        y_diff = np.column_stack([self.fish_xy[:,1]]*N) - np.row_stack([self.fish_xy[:,1]]*N) 

        v_x_diff = np.column_stack([self.fish_vel[:,0]]*N) - np.row_stack([self.fish_vel[:,0]]*N)
        v_y_diff = np.column_stack([self.fish_vel[:,1]]*N) - np.row_stack([self.fish_vel[:,1]]*N)
        
        ## Derived matricis ##
        distances = np.sqrt(x_diff**2 + y_diff**2)
        inv_distances = 1/(distances+0.000000001)
        neighbr_mat = self.neighbourhood(distances)

        vel_distances = np.sqrt(v_x_diff**2 + v_y_diff**2)
        inv_vel_distances = 1/(vel_distances+0.000000001)

        ## PREYS: ##
        # Prey to Prey: X & Y center of mass
        temp_matrix = neighbr_mat[n_preds:,n_preds:] * inv_distances[n_preds:, n_preds:]
        return_matrix[ n_preds:,0] = 1/(n_preys-1) * np.sum(temp_matrix * x_diff[n_preds:, n_preds:], axis=0)
        return_matrix[ n_preds:,1] = 1/(n_preys-1) * np.sum(temp_matrix * y_diff[n_preds:, n_preds:], axis=0)
        
        # Prey to prey: X & Y velocity:
        temp_matrix = neighbr_mat[n_preds:,n_preds:] * inv_vel_distances[n_preds:, n_preds:]
        return_matrix[ n_preds:,2] = 1/(n_preys-1) * np.sum(temp_matrix * v_x_diff[n_preds:, n_preds:], axis=0)
        return_matrix[ n_preds:,3] = 1/(n_preys-1) * np.sum(temp_matrix * v_y_diff[n_preds:, n_preds:], axis=0)

        ##TODO: # Prey-Pred: X & Y. center of mass
        temp_matrix = neighbr_mat[:n_preds,n_preds:] * inv_distances[:n_preds, n_preds:]
        return_matrix[n_preds:, 4] = (1/n_preds) * np.sum(temp_matrix * x_diff[:n_preds, n_preds:], axis=0)
        return_matrix[n_preds:, 5] = (1/n_preds) * np.sum(temp_matrix * y_diff[:n_preds, n_preds:], axis=0)
        
        ##TODO: # Prey-Pred: X & Y. velocity
        temp_matrix = neighbr_mat[:n_preds,n_preds:] * inv_vel_distances[:n_preds, n_preds:]
        return_matrix[n_preds:, 6] = (1/n_preds) * np.sum(temp_matrix * v_x_diff[:n_preds, n_preds:], axis=0)
        return_matrix[n_preds:, 7] = (1/n_preds) * np.sum(temp_matrix * v_y_diff[:n_preds, n_preds:], axis=0)
       
        ## PREDETORS ##
        # Pred-Pred: X & Y center of mass
        temp_matrix = neighbr_mat[:n_preds, :n_preds] * inv_distances[:n_preds, :n_preds]
        return_matrix[:n_preds, 0] = 1/(n_preds-1) * np.sum(temp_matrix * x_diff[:n_preds, :n_preds], axis=0)
        return_matrix[:n_preds, 1] = 1/(n_preds-1) * np.sum(temp_matrix * y_diff[:n_preds, :n_preds], axis=0)

        # Pred-Pred: X & Y velocity
        temp_matrix = neighbr_mat[:n_preds, :n_preds] * inv_vel_distances[:n_preds, :n_preds]
        return_matrix[:n_preds, 2] = 1/(n_preds-1) * np.sum(temp_matrix * v_x_diff[:n_preds, :n_preds], axis=0)
        return_matrix[:n_preds, 3] = 1/(n_preds-1) * np.sum(temp_matrix * v_y_diff[:n_preds, :n_preds], axis=0)

        # TODO: # Pred-Prey: X & Y. center of mass
        temp_matrix = neighbr_mat[n_preds:, :n_preds] * inv_distances[n_preds:, :n_preds]
        return_matrix[:n_preds, 4] = 1/n_preys * np.sum(temp_matrix * x_diff[n_preds:, :n_preds], axis=0)
        return_matrix[:n_preds, 5] = 1/n_preys * np.sum(temp_matrix * y_diff[n_preds:, :n_preds], axis=0)

        # TODO: # Pred-Prey: X & Y. velocity
        temp_matrix = neighbr_mat[n_preds:, :n_preds] * inv_vel_distances[n_preds:, :n_preds]
        return_matrix[:n_preds, 6] = 1/n_preys * np.sum(temp_matrix * v_x_diff[n_preds:, :n_preds], axis=0)
        return_matrix[:n_preds, 7] = 1/n_preys * np.sum(temp_matrix * v_y_diff[n_preds:, :n_preds], axis=0)

        # TODO: Relative position to wall. X & Y. [-1, 1]
        return_matrix[:, 8] = 2*self.fish_xy[:,0]/self.size_X-1
        return_matrix[:, 9] = 2*self.fish_xy[:,1]/self.size_Y-1

        return return_matrix


    def timestep(self,dt):
        
        self.brain_input = self.calculate_inputs()
        for i in self.interval_prey:
            acc_temp = self.prey_brain.make_decision(self.brain_input[i,:])
            norm_acc = np.linalg.norm(acc_temp)
            if norm_acc>1:
                self.acc_fish[i] = self.max_vel_prey * acc_temp / norm_acc
            else:
                self.acc_fish[i] = self.max_vel_prey * acc_temp

        for i in self.interval_pred:
            acc_temp = self.pred_brain.make_decision(self.brain_input[i, :])
            norm_acc = np.linalg.norm(acc_temp)
            if norm_acc > 1:
                self.acc_fish[i] = self.max_vel_pred * acc_temp / norm_acc
            else:
                self.acc_fish[i] = self.max_vel_pred * acc_temp


        # Integrate new position and velocity.
        self.fish_xy += self.fish_vel*dt + 0.5*self.acc_fish*dt*dt
        self.fish_vel += self.acc_fish*dt

        # Correct for reflective boundary
        for i in range(len(self.fish_xy)):
            if self.fish_xy[i, 0] < 0:
                self.fish_xy[i, 0] = 0
                self.fish_vel[i, 0] = 0
            elif self.fish_xy[i, 0] > self.size_X:
                self.fish_xy[i, 0] = self.size_X
                self.fish_vel[i, 0] = 0
            if self.fish_xy[i, 1] < 0:
                self.fish_xy[i, 1] = 0
                self.fish_vel[i, 1] = 0
            elif self.fish_xy[i, 1] > self.size_Y:
                self.fish_xy[i, 1] = self.size_Y
                self.fish_vel[i, 1] = 0

        # Correct for max velocities
        vel_magnitudes = np.linalg.norm(self.fish_vel,axis=1)
        for i in self.interval_prey:
            if vel_magnitudes[i] > self.max_vel_prey:
                self.fish_vel[i] = self.max_vel_prey * self.fish_vel[i] / vel_magnitudes[i]
        for i in self.interval_pred:
            if vel_magnitudes[i] > self.max_vel_pred:
                self.fish_vel[i] = self.max_vel_pred * self.fish_vel[i] / vel_magnitudes[i]


        # Check shark eats fish
        for shark in self.interval_pred:
            for prey in self.interval_prey:
                if prey >= len(self.fish_xy) or shark >= len(self.fish_xy):
                    #TODO: Good code shouldn't need this
                    break

                if self.eat_radius > np.linalg.norm(self.fish_xy[shark,:]-self.fish_xy[prey,:]):
                    self.eaten += 1
                    self.fish_xy    = np.delete(self.fish_xy, prey, axis=0)
                    self.fish_vel   = np.delete(self.fish_vel, prey, axis=0)
                    self.acc_fish   = np.delete(self.acc_fish, prey, axis=0)                   
                    self.interval_prey.pop()
                    break #A shark can only eat one fish per time step.

        #todo-future: # Correct for collision ???

    def set_videoutput(self, filename, fps=15, dpi=100):
        if self.video_enabled :
            raise BaseException("confusing to call set_videoutput() multiple times")

        self.video_enabled  = True

        FFMpeg_writer = manimation.writers['ffmpeg']
        metadata = dict(title='Fish in a tank simulation', artist='SoCS: Group 19',
                        comment='Very clever video if we may say so ourselves!')

        self.video_writer = FFMpeg_writer(fps=fps, metadata=metadata)
        self.fig = plt.figure()

        self.plot_ax = plt.gca()

        self.plot_prey, = plt.plot([], [], 'go', ms=5)
        self.plot_pred, = plt.plot([], [], 'ro', ms=5)
        self.plot_text = self.plot_ax.text(0,0.9, "Fish eaten = "+str(self.eaten))

        self.plot_prey_arrow, = plt.plot([], [], 'b-')
        self.plot_pred_arrow, = plt.plot([], [], 'k-')
        self.plot_prey_pred_arrow, = plt.plot([], [], 'g-')
        self.plot_pred_prey_arrow, = plt.plot([], [], 'r-')

        self.video_filename = filename
        self.video_dpi = dpi

        plt.xlim(-0.05, self.size_X*1.05)
        plt.ylim(-0.05, self.size_Y*1.05)
        title = "Aquarium"
        plt.title(title)

    def run_simulation(self):

        dt = 0.1 #todo: calculate dt from max vel and acc. hashtag physics
        time = 0
        MAX_TIME = 20
        HALF_NBR_FISHES = len(self.fish_xy_start) // 2

        self.fish_xy = np.copy(self.fish_xy_start )

        self.eaten = 0
        self.fish_vel = np.zeros(self.fish_xy_start.shape)
        self.acc_fish = np.zeros(self.fish_xy_start.shape)

        self.interval_pred = list(range(self.nbr_of_pred))
        self.interval_prey = list(range(self.nbr_of_pred, self.nbr_of_prey+self.nbr_of_pred))

        if self.video_enabled:
            with self.video_writer.saving(self.fig, self.video_filename, self.video_dpi):
                #TODO: Decide max_iterations
                while time < MAX_TIME and self.eaten <= HALF_NBR_FISHES:
                    self.timestep(dt)
                    self.__grab_Frame_()
                    time += dt
        else:
            while time < MAX_TIME and self.eaten <= HALF_NBR_FISHES:
                self.timestep(dt)
                time += dt

        score = self.eaten/time
        return (-score, score) #Prey score is negative pred score


    def __grab_Frame_(self):
        self.plot_prey.set_data(self.fish_xy[self.interval_prey,0], self.fish_xy[self.interval_prey,1])
        self.plot_pred.set_data(self.fish_xy[self.interval_pred,0], self.fish_xy[self.interval_pred,1])

        x_data_ss = [self.fish_xy[0, 0], self.fish_xy[0, 0] + self.brain_input[0, 0]]
        y_data_ss = [self.fish_xy[0, 1], self.fish_xy[0, 1] + self.brain_input[0, 1]]
        x_data_ff = [self.fish_xy[2, 0], self.fish_xy[2, 0] + self.brain_input[2, 0]]
        y_data_ff = [self.fish_xy[2, 1], self.fish_xy[2, 1] + self.brain_input[2, 1]]
        x_data_fs = [self.fish_xy[2, 0], self.fish_xy[2, 0] + self.brain_input[2, 4]]
        y_data_fs = [self.fish_xy[2, 1], self.fish_xy[2, 1] + self.brain_input[2, 5]]
        x_data_sf = [self.fish_xy[0, 0], self.fish_xy[0, 0] + self.brain_input[0, 4]]
        y_data_sf = [self.fish_xy[0, 1], self.fish_xy[0, 1] + self.brain_input[0, 5]]

        self.plot_pred_arrow.set_data(x_data_ss, y_data_ss)
        self.plot_prey_arrow.set_data(x_data_ff, y_data_ff)
        self.plot_pred_prey_arrow.set_data(x_data_sf, y_data_sf)
        self.plot_prey_pred_arrow.set_data(x_data_fs, y_data_fs)

        #print(x_data_sf)

        self.plot_text.set_text("Fish eaten = "+str(self.eaten))
        self.video_writer.grab_frame()

if __name__ == '__main__':

    aquarium_paramters = {'nbr_of_prey': 15, 'nbr_of_pred': 2, 'size_X': 1, 'size_Y': 1, 'max_speed_prey': 0.07,
                          'max_speed_pred': 0.1, 'max_acc_prey': 0.1, 'max_acc_pred': 0.1, 'eat_radius': 0.1,
                          'weight_range': 5, 'nbr_of_hidden_neurons': 10, 'nbr_of_inputs': 10, 'nbr_of_outputs': 2,
                          'visibility_range': 0.3}

    np.set_printoptions(precision=3)
    a = aquarium(**aquarium_paramters)
    a.set_videoutput('test.mp4')
    print(a.run_simulation())
