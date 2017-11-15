import numpy as np
random = np.random.random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation


class aquarium(object):
    # TODO:
    #  - kill simulations with useless sharks
    #  -

    def __init__(self, nbr_of_prey, nbr_of_pred, size_X, size_Y,
                 max_speed_prey,max_speed_pred, eat_radius):


        #ToDo Init object variables
        self.video_enabled = False
        self.size_X = size_X
        self.size_Y = size_Y

        self.pred_brain = None #To be inited later
        self.prey_brain = None #To be inited later
        self.eat_radius = eat_radius

        self.eaten = 0
        self.max_vel_prey = max_speed_prey
        self.max_vel_pred = max_speed_pred

        self.nbr_prey = nbr_of_prey
        self.nbr_pred = nbr_of_pred

        self.interval_prey = list(range(nbr_of_prey))
        self.interval_pred = list(range(nbr_of_prey,nbr_of_prey+nbr_of_pred))

        #Randomly place fishes and sharks
        self.fish_xy_start = np.copy()
        self.fish_xy = np.matrix(random(size=(nbr_of_prey+nbr_of_pred,2)))\
                       *np.matrix([[size_X,0],[0,size_Y]])

        self.acc_fish = np.zeros(fish_xy.shape)
        # Velocities are zeros to start with
        self.fish_vel = np.matrix(np.zeros(self.fish_xy.shape))


    def timestep(self,dt=1):
        #todo: # Get descisions for accelerations from brains.
        #todo: #Normalize to max_acceleration for prey and fish

        for i in self.interval_prey:

            brain_args = {"mean_prey_pos":None,
                          "mean_predator_pos":None,
                          "mean_prey_vel":None,
                          "mean_predator_vel":None,
                          "rel_pos":None}
            self.acc_fish[i] = self.prey_brain(**brain_args)

        # Todo: Write function to calculated weighted positions and velocities
        acc_fish = (random(self.fish_xy.shape) - 0.5) * 0.01


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


        #todo: # Check shark eats fish and update shark eating timer

        for shark in self.interval_pred:
            for prey in self.interval_prey:
                if eat_radius > np.linalg.norm(self.fish_xy[shark,:]-self.fish_xy[prey,:]):
                    self.eaten += 1
                    self.fish_xy[prey, :] = random((1, 2))
                    self.fish_vel[prey, :] = np.zeros((1, 2))


        #todo: # Correct for collision


        #todo: # Check shark starvation

        #todo: # Update timer: fish survival

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

    def run_simulation(fish_array):
        #TODO: Figure out what to do with these weights here (:S)

        with self.video_writer.saving(self.fig, self.video_filename, self.video_dpi):
            #TODO: Decide max_iterations
            for i in range(300):
                self.timestep(1)
                self.__grab_Frame_()



    def __grab_Frame_(self):
        self.plot_prey.set_data(self.fish_xy[self.interval_prey,0], self.fish_xy[self.interval_prey,1])
        self.plot_pred.set_data(self.fish_xy[self.interval_pred,0], self.fish_xy[self.interval_pred,1])
        self.plot_text.set_text("Fish eaten = "+str(self.eaten))
        self.video_writer.grab_frame()


raise NotImplementedError("Sorry, ledsen, förlåt: Vi gick hem trots ofärdig kod.")

a = aquarium(10,2,1,1,0.01,2)
a.set_videoutput("test.mp4")
a.run(None)