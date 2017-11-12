import numpy as np
random = np.random.random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation


class aquarium(object):
    def __init__(self, fishes, sharks, size_X, size_Y):

        #ToDo Init object variables
        self.video_enabled = False
        self.size_X = size_X
        self.size_Y = size_Y


        #Randomly place fishes and sharks
        self.fishes = np.matrix(random(size=(fishes,2)))*np.matrix([[size_X,0],[0,size_Y]])
        self.sharks = np.matrix(random(size=(sharks,2)))*np.matrix([[size_X,0],[0,size_Y]])

    def timestep(self):
        #todo: # Get descisions for accelerations from brains

        #todo: # Velocity verlet update for movements

        #todo: # Check shark eats fish and update shark eating timer

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

        self.plot_fishes, = plt.plot([], [], 'go', ms=5)
        self.plot_sharks, = plt.plot([], [], 'ro', ms=5)

        self.video_filename = filename
        self.video_dpi = dpi

        plt.xlim(-0.05, self.size_X*1.05)
        plt.ylim(-0.05, self.size_Y*1.05)

        title = "Aquarium"
        plt.title(title)

    def run(self):
        with self.video_writer.saving(self.fig, self.video_filename, self.video_dpi):
            for i in range(100):
                self.fishes += random(size=self.fishes.shape) * 0.01
                self.sharks += random(size=self.sharks.shape)*0.01
                self.__grab_Frame_()



    def __grab_Frame_(self):
        self.plot_fishes.set_data(self.fishes[:,0],self.fishes[:,1])
        self.plot_sharks.set_data(self.sharks[:,0],self.sharks[:,1])
        self.video_writer.grab_frame()



a = aquarium(10,1,1,1)
a.set_videoutput("test.mp4")
a.run()