# draggable rectangle with the animation blit techniques; see
# http://www.scipy.org/Cookbook/Matplotlib/Animations

import numpy as np
import matplotlib.pyplot as plt
random = np.random.random

from Aquarium import aquarium






class PlotHandler:
    def __init__(self,figure,aq_par):
        self.aq = aquarium(**aq_par)
        self.figure = figure

        self.ax = self.figure.add_subplot(111)

        self.press = None

        N = len(self.aq.fish_xy_start)

        # Fix the circles here
        self.circles = list()
        for i in range(N):
            pos = (self.aq.fish_xy_start[i,0], self.aq.fish_xy_start[i,1])
            self.circles.append(plt.Circle(pos,self.aq.size_Y/50))
            self.ax.add_patch(self.circles[i])

            if i in self.aq.interval_pred:
                self.circles[i].set_facecolor("r")
            elif i in self.aq.interval_prey:
                self.circles[i].set_facecolor("g")

        self.circle_pressed = [False]*(len(self.aq.fish_xy_start))

        # Setup the lines to show input vectors
        self.input_arrows = []
        for i in range(N):
            arrow_row = []
            for j in range(self.aq.pred_brain.nbr_of_inputs//2):
                line, = plt.plot([], [],'b')
                arrow_row.append(line)
                self.ax.add_line(line)
            self.input_arrows.append(arrow_row)


        self.ax.set_ylim([0,self.aq.size_Y])
        self.ax.set_xlim([0,self.aq.size_X])

        self.connect()

    def connect(self):
        # Connect to events of plot
        self.cidpress = self.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)
        self.update_arrows()
    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'

        if event.inaxes != self.ax: return


        click_on_circle = False
        for i,circ in reversed(list(enumerate(self.circles))):
            if circ.contains(event)[0]:
                click_on_circle = True
                break

        if not click_on_circle: return


        self.circle_pressed[i]=True

        x0, y0 = self.circles[i].center
        self.press = x0, y0, event.xdata, event.ydata
        self.update_arrows()

    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if not (True in self.circle_pressed): return
        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress

        index = np.where(self.circle_pressed)[0][0] #dont ask
        self.circles[index].center = (x0 + dx,y0 + dy)

        self.update_arrows()



    def on_release(self, event):
        'on release we reset the press data'
        self.circle_pressed=[False]*len(self.circle_pressed)
        self.update_arrows()
    def update_arrows(self):
        #Set fish_xy and calculate input
        for i,circ in enumerate(self.circles):
            self.aq.fish_xy[i,0] = circ.center[0]
            self.aq.fish_xy[i,1] = circ.center[1]

        inputs = self.aq.calculate_inputs()

        #update arrow that point
        for i,arrow_row in enumerate(self.input_arrows):
            for j,line in enumerate(arrow_row):
                x0 = self.circles[i].center[0]
                y0 = self.circles[i].center[1]
                dx = inputs[i,j*2]
                dy = inputs[i,j*2+1]
                line.set_data([x0,x0+dx], [y0,y0+dy])

        #draw
        self.figure.canvas.draw()

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.figure.canvas.mpl_disconnect(self.cidpress)
        self.figure.canvas.mpl_disconnect(self.cidrelease)
        self.figure.canvas.mpl_disconnect(self.cidmotion)




fig = plt.figure()
aquarium_paramters = {'nbr_of_prey': None, 'nbr_of_pred': None, 'size_X': None, 'size_Y': None, 'max_speed_prey': 0.07,
                          'max_speed_pred': 0.1, 'max_acc_prey': 0.1, 'max_acc_pred': 0.1, 'eat_radius': 0.1,
                          'weight_range': 5, 'nbr_of_hidden_neurons': 10, 'nbr_of_outputs': 2,
                          'visibility_range': 1.5, 'input_set': None }
aquarium_paramters["nbr_of_prey"] = 4
aquarium_paramters["nbr_of_pred"] = 2
aquarium_paramters["size_X"] = 4
aquarium_paramters["size_Y"] = 4
aquarium_paramters["input_set"] = ["enemy_pos"]


ph = PlotHandler(fig,aquarium_paramters)

print("Let the Show begin")
plt.show()
