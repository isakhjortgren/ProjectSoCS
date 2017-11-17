import numpy as np
import numpy.matlib
random = np.random.random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

from Brain import Brain


class aquarium(object):
    # TODO:
    #  - kill simulations with useless sharks (For future!)
    #  - calculate dt with respect to max_acc and max_vel

    def __init__(self, nbr_of_prey, nbr_of_pred, size_X, size_Y,
                 max_speed_prey,max_speed_pred,max_acc_prey,max_acc_pred, 
                 eat_radius, nbr_of_hidden_neurons,nbr_of_inputs,nbr_of_outputs,
                 weight_range ):


        #ToDo Init object variables
        self.video_enabled = False
        self.size_X = size_X
        self.size_Y = size_Y

        self.pred_brain = Brain(nbr_of_hidden_neurons, nbr_of_inputs, nbr_of_outputs, weight_range) 
        self.prey_brain = Brain(nbr_of_hidden_neurons, nbr_of_inputs, nbr_of_outputs, weight_range) 
        
        self.eat_radius = eat_radius

        self.eaten = 0
        
        self.max_vel_prey = max_speed_prey
        self.max_vel_pred = max_speed_pred
        
        self.max_acc_prey = max_acc_prey
        self.max_acc_pred = max_acc_pred

        self.nbr_prey = nbr_of_prey
        self.nbr_pred = nbr_of_pred

        self.interval_prey = list(range(nbr_of_prey))
        self.interval_pred = list(range(nbr_of_prey,nbr_of_prey+nbr_of_pred))

        #Constant
        self.fish_xy_start = np.matrix(random(size=(nbr_of_prey+nbr_of_pred,2)))\
                            *np.matrix([[size_X,0],[0,size_Y]])
        
        self.fish_xy = np.matlib.zeros(fish_xy_start.shape)
        self.fish_vel = np.matlib.zeros(fish_xy_start.shape)
        self.acc_fish = np.matlib.zeros(fish_xy_start.shape)

    def calculate_inputs(self):
        N = len(fish_xy)
        # Position differences
        x_diff = np.column_stack([self.fish_xy[:,0]]*N) - np.row_stack([self.fish_xy[:,0]]*N) 
        y_diff = np.column_stack([self.fish_xy[:,1]]*N) - np.row_stack([self.fish_xy[:,1]]*N) 

        # Velocity differences
        v_x_diff = np.column_stack([self.fish_vel[:,0]]*N) - np.row_stack([self.fish_vel[:,0]]*N) 
        v_y_diff = np.column_stack([self.fish_vel[:,1]]*N) - np.row_stack([self.fish_vel[:,1]]*N) 
        
        

    def timestep(self,dt):
        #todo: # Get descisions for accelerations from brains.
        

        for i in self.interval_prey:

            brain_args = {"mean_prey_pos":None,
                          "mean_predator_pos":None,
                          "mean_prey_vel":None,
                          "mean_predator_vel":None,
                          "rel_pos":None}
            self.acc_fish[i] = self.prey_brain(**brain_args)
        



        #Normalize to max_acceleration for prey and fish
        acc_magnitudes = np.linalg.norm(self.acc_fish,axis=1)
        for i in self.interval_prey:
            if acc_magnitudes[i] > self.max_acc_prey:
                acc_fish[i] = self.max_acc_prey * acc_fish[i] / acc_magnitudes[i]
        for i in self.interval_pred:
            if acc_magnitudes[i] > self.max_acc_pred:
                acc_fish[i] = self.max_acc_pred * acc_fish[i] / acc_magnitudes[i]


        # Integrate new position and velocity.
        self.fish_xy += self.fish_vel*dt + 0.5*acc_fish*dt*dt
        self.fish_vel += acc_fish*dt

        # Correct for max velocities
        vel_magnitudes = np.linalg.norm(self.fish_vel,axis=1)
        for i in self.interval_prey:
            if vel_magnitudes[i] > self.max_vel_prey:
                self.fish_vel[i] = self.max_vel_prey * self.fish_vel[i] / vel_magnitudes[i]
        for i in self.interval_pred:
            if vel_magnitudes[i] > self.max_vel_pred:
                self.fish_vel[i] = self.max_vel_pred * self.fish_vel[i] / vel_magnitudes[i]

        #Correct for reflective boundary
        for i in range(len(self.fish_xy)):
            if self.fish_xy[i, 0] < 0:
                self.fish_xy[i, 0] = 0
                self.fish_vel[i,0] = 0
            elif self.fish_xy[i, 0] > self.size_X:
                self.fish_xy[i, 0] = self.size_X
                self.fish_vel[i, 0] = 0
            if self.fish_xy[i, 1] < 0:
                self.fish_xy[i, 1] = 0
                self.fish_vel[i, 1] = 0
            elif self.fish_xy[i, 1] > self.size_Y:
                self.fish_xy[i, 1] = self.size_Y
                self.fish_vel[i, 1] = 0


        #todo future: # Check shark eats fish and update shark eating timer

        for shark in self.interval_pred:
            for prey in self.interval_prey:
                if eat_radius > np.linalg.norm(self.fish_xy[shark,:]-self.fish_xy[prey,:]):
                    self.eaten += 1
                    self.fish_xy[prey, :] = random((1, 2))
                    self.fish_vel[prey, :] = np.zeros((1, 2))


        #todo: # Correct for collision


        #todo future: # Check shark starvation

        #todo future: # Update timer: fish survival

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


        self.video_filename = filename
        self.video_dpi = dpi

        plt.xlim(-0.05, self.size_X*1.05)
        plt.ylim(-0.05, self.size_Y*1.05)

        title = "Aquarium"
        plt.title(title)

    def run_simulation():

        dt = 1 #todo: calculate dt from max vel and acc. hashtag physics
        time = 0
        MAX_TIME = 20
        HALF_NBR_FISHES = self.nbr_prey // 2

        self.fish_xy = np.copy(self.fish_xy_start )
        
        self.fish_vel.fill(0)
        self.acc_fish.fill(0)


        with self.video_writer.saving(self.fig, self.video_filename, self.video_dpi):
            #TODO: Decide max_iterations
            while time < MAX_TIME and self.eaten <= HALF_NBR_FISHES
                self.timestep(dt)
                self.__grab_Frame_()

                time += dt

        score = self.eaten/time
        return (-score, score) #Prey score is negative pred score


    def __grab_Frame_(self):
        self.plot_prey.set_data(self.fish_xy[self.interval_prey,0], self.fish_xy[self.interval_prey,1])
        self.plot_pred.set_data(self.fish_xy[self.interval_pred,0], self.fish_xy[self.interval_pred,1])
        self.plot_text.set_text("Fish eaten = "+str(self.eaten))
        self.video_writer.grab_frame()


raise NotImplementedError("Sorry, ledsen, förlåt: Vi gick hem trots ofärdig kod.")

a = aquarium(10,2,1,1,0.01,2)
a.set_videoutput("test.mp4")
a.run(None)