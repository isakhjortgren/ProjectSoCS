import numpy as np
import numpy.matlib
random = np.random.random

import matplotlib
import time

import math 
import time

import matplotlib.pyplot as plt
try:
    import matplotlib.animation as manimation
except ImportError:
    pass

from Brain import Brain, randomBrain, attackBrain,dodgeBrain


class aquarium(object):
    # TODO:
    #  - kill simulations with useless sharks (For future!)
    #  - calculate dt with respect to eat_radius and max_speeds

    def __init__(self, nbr_of_prey, nbr_of_pred, size_X, size_Y,
                 max_speed_prey,max_speed_pred,max_acc_prey,max_acc_pred, 
                 eat_radius, nbr_of_hidden_neurons,nbr_of_outputs,
                 weight_range, visibility_range, safe_boundary=True, input_type='closest',
                 rand_walk_brain_set=[], input_set=["friend_pos","friend_vel","enemy_pos","enemy_vel","wall"]):

        #ToDo Init object variables
        self.record_video = False
        self.video_enabled = False
        self.size_X = size_X
        self.size_Y = size_Y

        nbr_of_inputs = len(input_set)*2
        self.inputs = input_set

        if "prey" in rand_walk_brain_set:
            self.prey_brain = dodgeBrain(nbr_of_hidden_neurons, nbr_of_inputs, nbr_of_outputs, weight_range)
        else: 
            self.prey_brain = Brain(nbr_of_hidden_neurons, nbr_of_inputs, nbr_of_outputs, weight_range)

        if "pred" in rand_walk_brain_set:
            self.pred_brain = attackBrain(nbr_of_hidden_neurons, nbr_of_inputs, nbr_of_outputs, weight_range)
        else:
            self.pred_brain = Brain(nbr_of_hidden_neurons, nbr_of_inputs, nbr_of_outputs, weight_range)


        self.eat_radius = eat_radius
        self.visibility_range = visibility_range

        self.nbr_of_prey = nbr_of_prey
        self.nbr_of_pred = nbr_of_pred

        self.max_vel_prey = max_speed_prey
        self.max_vel_pred = max_speed_pred
        
        self.max_acc_prey = max_acc_prey
        self.max_acc_pred = max_acc_pred

        self.collision_len = 0.5*self.eat_radius
        self.safe_boundary = safe_boundary



        #Constant
        self.fish_xy_start = np.matrix(random(size=(nbr_of_prey+nbr_of_pred,2)))\
                            *np.matrix([[size_X, 0], [0,size_Y]])
        
        self.brain_input = None 

        self.interval_pred = list(range(self.nbr_of_pred))
        self.interval_prey = list(range(self.nbr_of_pred, self.nbr_of_prey + self.nbr_of_pred))

        self.eaten = None

        self.fish_xy = np.copy(self.fish_xy_start)
        self.fish_vel = np.zeros(self.fish_xy_start.shape)
        self.acc_fish = np.zeros(self.fish_xy_start.shape)

        self.x_diff = None
        self.y_diff = None
        self.pred_score = None
        self.prey_score = None
        self.MAX_TIME = None
        self.max_vels = None

        self.time_last_snack = None

        self.rare_bug_counter = None

        if input_type == 'closest':
            self.calculate_inputs = self.calculate_inputs_closest
            self.neighbourhood = self.neighbourhood_closest
        elif input_type == 'weighted':
            self.calculate_inputs = self.calculate_inputs_weighted
            self.neighbourhood = self.neighbourhood_weighted
        else:
            raise ValueError('Use a proper input type you idiot!')

    def neighbourhood_closest(self, distances):
        return np.exp(-distances**2/(2*self.visibility_range**2)) 

    def calculate_inputs_closest(self):
        
        next_col = 0

        n_preds = len(self.interval_pred)
        n_preys = len(self.interval_prey)

        N = len(self.fish_xy) # also equal to (n_preds + n_preys)

        return_matrix = np.zeros(( N, self.pred_brain.nbr_of_inputs))

        ## Differences ##
        self.x_diff = np.column_stack([self.fish_xy[:,0]]*N) - np.row_stack([self.fish_xy[:,0]]*N) 
        self.y_diff = np.column_stack([self.fish_xy[:,1]]*N) - np.row_stack([self.fish_xy[:,1]]*N) 

        x_diff = self.x_diff 
        y_diff = self.y_diff

        ## Derived matricis ##
        distances = np.sqrt(x_diff**2 + y_diff**2)
        inv_distances = 1/(distances+0.000000001)
        neighbr_mat = self.neighbourhood(distances) * inv_distances
      
        velocity_normation_factor = 1/(self.max_vel_pred + self.max_vel_prey)

        if "friend_pos" in self.inputs:
            # Prey to Prey: X & Y center of mass
            closest = np.argmin(distances[n_preds:, n_preds:]+np.identity(n_preys)*100,axis=1)

            i_es = list(range(n_preds, n_preds+n_preys))
            j_es = closest+n_preds

            return_matrix[n_preds: , next_col ]     = -x_diff[i_es,j_es ] * neighbr_mat[i_es,j_es ]
            return_matrix[n_preds: , next_col +1]   = -y_diff[i_es,j_es ] * neighbr_mat[i_es,j_es ]

            closest = np.argmin(distances[:n_preds,:n_preds]+np.identity(n_preds)*100,axis=1)
            i_es = list(range(n_preds))
            j_es = closest
            
            return_matrix[:n_preds , next_col ]     = -x_diff[ i_es, j_es] * neighbr_mat[i_es,j_es ]
            return_matrix[:n_preds , next_col +1]   = -y_diff[ i_es, j_es] * neighbr_mat[i_es,j_es ]

            next_col += 2

        if "enemy_pos" in self.inputs: 

            # Prey -> Pred
            closest = np.argmin(distances[n_preds:, :n_preds] ,axis=1)

            i_es = list(range(n_preds, n_preds+n_preys))
            j_es = closest
            return_matrix[n_preds:, next_col]       = -x_diff[i_es, j_es]* neighbr_mat[i_es,j_es ]
            return_matrix[n_preds:, next_col + 1]   = -y_diff[i_es, j_es]* neighbr_mat[i_es,j_es ]

            # Pred -> Prey
            closest = np.argmin(distances[:n_preds, n_preds:] ,axis=1)

            i_es = list(range(n_preds))
            j_es = closest+n_preds

            return_matrix[:n_preds, next_col]       = -x_diff[i_es, j_es]* neighbr_mat[i_es,j_es ]
            return_matrix[:n_preds, next_col + 1]   = -y_diff[i_es, j_es]* neighbr_mat[i_es,j_es ]

            next_col += 2

        if "enemy_vel" in self.inputs:
            v_x_diff = np.column_stack([self.fish_vel[:, 0]] * N) - np.row_stack([self.fish_vel[:, 0]] * N)
            v_y_diff = np.column_stack([self.fish_vel[:, 1]] * N) - np.row_stack([self.fish_vel[:, 1]] * N)

            # Prey -> Pred
            closest = np.argmin(distances[n_preds:, :n_preds] ,axis=1)

            i_es = list(range(n_preds, n_preds+n_preys))
            j_es = closest
            return_matrix[n_preds:, next_col]       = -v_x_diff[i_es, j_es] * velocity_normation_factor
            return_matrix[n_preds:, next_col + 1]   = -v_y_diff[i_es, j_es] * velocity_normation_factor

             # Pred -> Prey
            closest = np.argmin(distances[:n_preds, n_preds:] ,axis=1)

            i_es = list(range(n_preds))
            j_es = closest+n_preds
            return_matrix[:n_preds, next_col]       = -v_x_diff[i_es, j_es] * velocity_normation_factor
            return_matrix[:n_preds, next_col + 1]   = -v_y_diff[i_es, j_es] * velocity_normation_factor

        if "wall" in self.inputs:
            #Relative position to wall. X & Y. [-1, 1]
            return_matrix[:, next_col] = 2*self.fish_xy[:,0]/self.size_X-1
            return_matrix[:, next_col+1] = 2*self.fish_xy[:,1]/self.size_Y-1

        return return_matrix

    def neighbourhood_weighted(self, distances):
        return np.exp(-distances ** 2 / (2 * self.visibility_range ** 2)) / self.visibility_range

    def calculate_inputs_weighted(self):
        next_col = 0

        n_preds = len(self.interval_pred)
        n_preys = len(self.interval_prey)

        N = len(self.fish_xy)  # also equal to (n_preds + n_preys)

        return_matrix = np.zeros((N, self.pred_brain.nbr_of_inputs))

        ## Differences ##
        self.x_diff = np.column_stack([self.fish_xy[:, 0]] * N) - np.row_stack([self.fish_xy[:, 0]] * N)
        self.y_diff = np.column_stack([self.fish_xy[:, 1]] * N) - np.row_stack([self.fish_xy[:, 1]] * N)

        x_diff = self.x_diff
        y_diff = self.y_diff

        ## Derived matricis ##
        distances = np.sqrt(x_diff ** 2 + y_diff ** 2)
        inv_distances = 1 / (distances + 0.000000001)
        neighbr_mat = self.neighbourhood(distances)

        if "friend_vel" in self.inputs or "enemy_vel" in self.inputs:
            v_x_diff = np.column_stack([self.fish_vel[:, 0]] * N) - np.row_stack([self.fish_vel[:, 0]] * N)
            v_y_diff = np.column_stack([self.fish_vel[:, 1]] * N) - np.row_stack([self.fish_vel[:, 1]] * N)

            vel_distances = np.sqrt(v_x_diff ** 2 + v_y_diff ** 2)
            inv_vel_distances = 1 / (vel_distances + 0.000000001)

        ## PREYS: ##
        if "friend_pos" in self.inputs:
            # Prey to Prey: X & Y center of mass
            temp_matrix = neighbr_mat[n_preds:, n_preds:] * inv_distances[n_preds:, n_preds:]
            return_matrix[n_preds:, next_col] = 1 / (n_preys - 1) * np.sum(temp_matrix * x_diff[n_preds:, n_preds:],
                                                                           axis=0)
            return_matrix[n_preds:, next_col + 1] = 1 / (n_preys - 1) * np.sum(temp_matrix * y_diff[n_preds:, n_preds:],
                                                                               axis=0)

            # Pred-Pred: X & Y center of mass
            temp_matrix = neighbr_mat[:n_preds, :n_preds] * inv_distances[:n_preds, :n_preds]
            return_matrix[:n_preds, next_col] = 1 / (n_preds - 1) * np.sum(temp_matrix * x_diff[:n_preds, :n_preds],
                                                                           axis=0)
            return_matrix[:n_preds, next_col + 1] = 1 / (n_preds - 1) * np.sum(temp_matrix * y_diff[:n_preds, :n_preds],
                                                                               axis=0)

            next_col += 2

        if "friend_vel" in self.inputs:
            # Prey to prey: X & Y velocity:
            temp_matrix = neighbr_mat[n_preds:, n_preds:] * inv_vel_distances[n_preds:, n_preds:]
            return_matrix[n_preds:, next_col] = 1 / (n_preys - 1) * np.sum(temp_matrix * v_x_diff[n_preds:, n_preds:],
                                                                           axis=0)
            return_matrix[n_preds:, next_col + 1] = 1 / (n_preys - 1) * np.sum(
                temp_matrix * v_y_diff[n_preds:, n_preds:],
                axis=0)

            # Pred-Pred: X & Y velocity
            temp_matrix = neighbr_mat[:n_preds, :n_preds] * inv_vel_distances[:n_preds, :n_preds]
            return_matrix[:n_preds, next_col] = 1 / (n_preds - 1) * np.sum(temp_matrix * v_x_diff[:n_preds, :n_preds],
                                                                           axis=0)
            return_matrix[:n_preds, next_col + 1] = 1 / (n_preds - 1) * np.sum(
                temp_matrix * v_y_diff[:n_preds, :n_preds],
                axis=0)

            next_col += 2

        if "enemy_pos" in self.inputs:
            # Prey-Pred: X & Y. center of mass
            temp_matrix = neighbr_mat[:n_preds, n_preds:] * inv_distances[:n_preds, n_preds:]
            return_matrix[n_preds:, next_col] = (1 / n_preds) * np.sum(temp_matrix * x_diff[:n_preds, n_preds:], axis=0)
            return_matrix[n_preds:, next_col + 1] = (1 / n_preds) * np.sum(temp_matrix * y_diff[:n_preds, n_preds:],
                                                                           axis=0)

            # Pred-Prey: X & Y. center of mass
            temp_matrix = neighbr_mat[n_preds:, :n_preds] * inv_distances[n_preds:, :n_preds]
            return_matrix[:n_preds, next_col] = 1 / n_preys * np.sum(temp_matrix * x_diff[n_preds:, :n_preds], axis=0)
            return_matrix[:n_preds, next_col + 1] = 1 / n_preys * np.sum(temp_matrix * y_diff[n_preds:, :n_preds],
                                                                         axis=0)

            next_col += 2

        ## PREDETORS ##
        if "enemy_vel" in self.inputs:
            # Pred-Prey: X & Y. velocity
            temp_matrix = neighbr_mat[n_preds:, :n_preds] * inv_vel_distances[n_preds:, :n_preds]
            return_matrix[:n_preds, next_col] = 1 / n_preys * np.sum(temp_matrix * v_x_diff[n_preds:, :n_preds], axis=0)
            return_matrix[:n_preds, next_col + 1] = 1 / n_preys * np.sum(temp_matrix * v_y_diff[n_preds:, :n_preds],
                                                                         axis=0)

            # Prey-Pred: X & Y. velocity
            temp_matrix = neighbr_mat[:n_preds, n_preds:] * inv_vel_distances[:n_preds, n_preds:]
            return_matrix[n_preds:, next_col] = (1 / n_preds) * np.sum(temp_matrix * v_x_diff[:n_preds, n_preds:],
                                                                       axis=0)
            return_matrix[n_preds:, next_col + 1] = (1 / n_preds) * np.sum(temp_matrix * v_y_diff[:n_preds, n_preds:],
                                                                           axis=0)

            next_col += 2

        if "wall" in self.inputs:
            # Relative position to wall. X & Y. [-1, 1]
            return_matrix[:, next_col] = 2 * self.fish_xy[:, 0] / self.size_X - 1
            return_matrix[:, next_col + 1] = 2 * self.fish_xy[:, 1] / self.size_Y - 1


        return return_matrix

    def timestep(self, dt, time):
        
        self.brain_input = self.calculate_inputs()
        
        #Get descisions and correct for max acceleration #TODO make faster with matrix operation. use map()
        self.acc_fish[self.interval_pred] = np.array(list(map(self.pred_brain.make_decision, self.brain_input[self.interval_pred])))
        self.acc_fish[self.interval_prey] = np.array(list(map(self.prey_brain.make_decision, self.brain_input[self.interval_prey])))
        
        acc_norm = np.linalg.norm(self.acc_fish,axis=1)
        
        self.acc_fish *= self.max_acc[:,np.newaxis]
        
        indices = np.where(acc_norm>1)
        if len(indices)>0:
            self.acc_fish[indices] =  self.acc_fish[indices] / acc_norm[indices,np.newaxis]

        # Take care of collisions
        N = len(self.fish_xy)
        x_diff = self.x_diff #Already calculated in calculate_inputs()
        y_diff = self.y_diff #Already calculated in calculate_inputs()
        collision_indicies =    (abs(x_diff)<self.collision_len) & \
                                (abs(y_diff)<self.collision_len) & \
                                np.tril(np.ones((N,N),dtype=bool),k=-1)

        collision_indicies = np.column_stack(np.where(collision_indicies))

        if len(collision_indicies)>0:
            i_es = collision_indicies[:,0]
            j_es = collision_indicies[:,1]

            col_vec = self.fish_xy[i_es,:]-self.fish_xy[j_es,:]
            distances = np.sqrt(np.sum(col_vec**2,axis=1))

            strength = 10*self.max_acc_prey - (10*self.max_acc_prey/self.collision_len)*distances
            
            strength[strength>self.max_acc_prey*5 ] = self.max_acc_prey*5
            strength[strength<0] = 0

            distances[distances<0.00000001] = 0.00000001 

            self.acc_fish[i_es,:] += strength[:,np.newaxis] * col_vec / distances[:,np.newaxis] 
            self.acc_fish[j_es,:] -= strength[:,np.newaxis] * col_vec / distances[:,np.newaxis] 
        
            
        # Integrate new position and velocity.
        self.fish_xy += self.fish_vel*dt + 0.5*self.acc_fish*dt*dt
        self.fish_vel += self.acc_fish*dt 

        # Effect of boundary, fish either stops or dies
        if self.safe_boundary:
            # Safe boundary, need to correct for reflective boundary
            indices = np.where(self.fish_xy[:,0] < 0)
            self.fish_xy[indices,0] =   random(len(indices)) / (self.size_X * 500)
            self.fish_vel[indices,0] =  0

            indices = np.where(self.fish_xy[:,0] > self.size_X)
            self.fish_xy[indices,0] =   self.size_X - random(len(indices)) / (self.size_X * 500)
            self.fish_vel[indices,0] =  0

            indices = np.where(self.fish_xy[:,1] < 0)
            self.fish_xy[indices,1] =   random(len(indices)) / (self.size_Y * 500)
            self.fish_vel[indices,1] =  0

            indices = np.where(self.fish_xy[:,1] > self.size_Y)
            self.fish_xy[indices,1] =   self.size_X - random(len(indices)) / (self.size_Y * 500)
            self.fish_vel[indices,1] =  0
        else:
            # Sharks stop at boundary, fishes die
            for i in self.interval_pred:
                if self.fish_xy[i, 0] < 0:
                    self.fish_xy[i, 0] = (random()) / (self.size_X * 500)
                    self.fish_vel[i, 0] = 0
                elif self.fish_xy[i, 0] > self.size_X:
                    self.fish_xy[i, 0] = self.size_X - (random()) / (self.size_X * 1000)
                    self.fish_vel[i, 0] = 0
                if self.fish_xy[i, 1] < 0:
                    self.fish_xy[i, 1] = (random()) / (self.size_Y * 500)
                    self.fish_vel[i, 1] = 0
                elif self.fish_xy[i, 1] > self.size_Y:
                    self.fish_xy[i, 1] = self.size_Y - (random()) / (self.size_Y * 1000)
                    self.fish_vel[i, 1] = 0

            off_boundary = (self.size_X < self.fish_xy[:, 0]) | (self.fish_xy[:, 0] < 0) | \
                           (self.size_Y < self.fish_xy[:, 1]) | (self.fish_xy[:, 1] < 0)
            off_boundary[self.interval_pred] = False  # don't kill sharks
            if True in off_boundary:
                off_boundary, = np.where(off_boundary)
                self.remove_fishes(off_boundary)
                self.prey_score -= len(off_boundary) / (time + self.MAX_TIME)

        # Check shark eats fish
        n_preds = len(self.interval_pred)
        n_preys = len(self.interval_prey)
        
        x_diff = np.column_stack([self.fish_xy[n_preds:,0]]*n_preds) - np.row_stack([self.fish_xy[:n_preds,0]]*n_preys) 
        y_diff = np.column_stack([self.fish_xy[n_preds:,1]]*n_preds) - np.row_stack([self.fish_xy[:n_preds,1]]*n_preys)

        eaten_indicies =    (abs(x_diff)<self.eat_radius) & (abs(y_diff)<self.eat_radius) 
        if True in eaten_indicies:
            eaten_indicies = np.column_stack(np.where(eaten_indicies))
            indices_of_eaten_fish = eaten_indicies[:,0]+n_preds
            self.remove_fishes(indices_of_eaten_fish)
            self.pred_score += len(indices_of_eaten_fish)/(time + self.MAX_TIME)
            self.time_last_snack = time
            self.prey_score -= len(indices_of_eaten_fish)/(time + self.MAX_TIME)
  
        # Correct for max velocities 
        vel_magnitudes = np.linalg.norm(self.fish_vel,axis=1)
        indices = vel_magnitudes>self.max_vels
        if True in indices:          
            self.fish_vel[indices,:] =  self.max_vels[indices,np.newaxis]  \
                                        * self.fish_vel[indices,:] \
                                        / vel_magnitudes[indices,np.newaxis]         

        #Simply error crashes with NaN-values
        if math.isnan(self.fish_xy[0,0]):
                raise RuntimeError("NaN value in coordinate")


    def set_videoutput(self, filename, fps=50, dpi=100):
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
        self.plot_text = self.plot_ax.text(0,0.9, "Fish killed = "+str(self.eaten))

        self.input_to_plot = len(self.inputs)
        if "wall" in self.inputs:
            self.input_to_plot -= 1

        self.plot_prey_arrow = []
        self.plot_pred_arrow = []
            
        for i in range(self.input_to_plot):
            temp, = plt.plot([], [], 'b-')
            self.plot_prey_arrow.append(temp)

            temp, = plt.plot([], [], 'r-')
            self.plot_pred_arrow.append(temp)


        self.fish_acc_arrow = list() 
        for i in range(self.fish_xy_start.shape[0]):
            temp, = plt.plot([], [], 'k--')
            self.fish_acc_arrow.append(temp)

        self.video_filename = filename
        self.video_dpi = dpi

        plt.xlim(-0.05, self.size_X*1.05)
        plt.ylim(-0.05, self.size_Y*1.05)
        title = "Aquarium"
        plt.title(title)

    def run_simulation(self):

        dt = 0.25*  min(self.eat_radius, self.collision_len) / max(self.max_vel_prey, self.max_vel_pred)
        time = 0

        self.MAX_TIME = 100
        self.MAX_TIME_SINCE_SNACK = 20
        self.rare_bug_counter = 0
        HALF_NBR_FISHES = len(self.fish_xy_start) // 2

        self.fish_xy = np.copy(self.fish_xy_start )

        self.eaten = 0
        self.fish_vel = np.zeros(self.fish_xy_start.shape)
        self.acc_fish = np.zeros(self.fish_xy_start.shape)

        self.pred_score = 0
        self.prey_score = 0
        self.interval_pred = list(range(self.nbr_of_pred))
        self.interval_prey = list(range(self.nbr_of_pred, self.nbr_of_prey+self.nbr_of_pred))

        self.max_vels = np.concatenate((\
                            np.ones(self.nbr_of_pred)*self.max_vel_pred, \
                            np.ones(self.nbr_of_prey)*self.max_vel_prey))

        self.max_acc = np.concatenate((\
                            np.ones(self.nbr_of_pred)*self.max_acc_pred, \
                            np.ones(self.nbr_of_prey)*self.max_acc_prey))


        self.time_last_snack = 0
        next_print = 1
        if self.video_enabled:
            with self.video_writer.saving(self.fig, self.video_filename, self.video_dpi):
                while time < self.MAX_TIME and self.eaten <= HALF_NBR_FISHES and time-self.time_last_snack<self.MAX_TIME_SINCE_SNACK:
                    self.timestep(dt, time)
                    self.__grab_Frame_()
                    time += dt
                    if time>next_print:
                        next_print += 1
                        print(time, "eaten =",self.eaten)
                     
        else:
            while time < self.MAX_TIME and self.eaten <= HALF_NBR_FISHES and time-self.time_last_snack<self.MAX_TIME_SINCE_SNACK:
                self.timestep(dt, time)
                time += dt

        return (self.prey_score, self.pred_score) #Prey score is negative pred score


    def __grab_Frame_(self):
        self.plot_prey.set_data(self.fish_xy[self.interval_prey,0], self.fish_xy[self.interval_prey,1])
        self.plot_pred.set_data(self.fish_xy[self.interval_pred,0], self.fish_xy[self.interval_pred,1])

        for i in range(self.fish_xy.shape[0]):
            x,y = self.fish_xy[i,:]
            #dx,dy = self.acc_fish[i,:]
            dx,dy = self.inputs[i,4:6]
            
            self.fish_acc_arrow[i].set_data([x,x+dx],[y,y+dy])

        self.plot_text.set_text("Fish killed = "+str(self.eaten))
        self.video_writer.grab_frame()
    
    def remove_fishes(self,indices):
        indices = list(set(indices))

        self.fish_xy  = np.delete(self.fish_xy,  indices, axis=0)
        self.fish_vel = np.delete(self.fish_vel, indices, axis=0)
        self.acc_fish = np.delete(self.acc_fish, indices, axis=0)
        self.max_vels = np.delete(self.max_vels, indices, axis=0)
        self.max_acc  = np.delete(self.max_acc,  indices, axis=0)

        self.eaten += len(indices)
        for i in range(len(indices)):
            if len(self.interval_prey)>0:
                self.interval_prey.pop()


if __name__ == '__main__':

    aquarium_parameters = {'nbr_of_prey': 20, 'nbr_of_pred': 2, 'size_X': 2, 'size_Y': 2, 
                        'max_speed_prey': 0.15,
                        'max_speed_pred': 0.2, 
                        'max_acc_prey': 0.3, 
                        'max_acc_pred': 0.15, 
                        'eat_radius': 0.05,
                       'weight_range': 1, 'nbr_of_hidden_neurons': 5, 'nbr_of_outputs': 2,
                       'visibility_range': 0.5, 'rand_walk_brain_set': [], 'input_type': 'closest',
                       'input_set': ["enemy_pos", "friend_pos", "enemy_vel","wall"], 'safe_boundary': True}

    np.set_printoptions(precision=3)
    a = aquarium(**aquarium_parameters)
    #a.set_videoutput('test.mp4',fps=25)
    start_time = time.time()
    print(a.run_simulation())

    print("Done in: ", round(time.time()-start_time,3), "s")

