import numpy as np
random = np.random.random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation


class aquarium(object):
    def __init__(self, fishes, sharks, size, grid_size):
        pass

    # Randomly place fishes and sharks



    def set_videoutput(self, filename, fps=15, dpi=100):
        if self.video_enabled:
            raise BaseException("confusing to call set_videoutput() multiple times")

        self.video_enabled = True

        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Complex Systems hw1', artist='Matplotlib',
                        comment='Beautiful video!')

        self.writer = FFMpegWriter(fps=fps, metadata=metadata)
        self.fig = plt.figure()

        self.plotTrees, = plt.plot([], [], 'go', ms=5)
        self.plotFires, = plt.plot([], [], 'ro', ms=5)

        self.video_filename = filename
        self.video_dpi = dpi

        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)

        title = self.__get_title_str()
        plt.title(title)

    def __grab_Frame_(self):
        size_inv = 1 / self.size

        firePos = []
        treePos = []

        add_fire = firePos.append
        add_tree = treePos.append
        for i, j in loop_range:
            if trees[i, j] == -1:
                trees[i, j] = 0
                add_fire([i * size_inv, j * size_inv])
            elif trees[i, j] == 1:
                add_tree([i * size_inv, j * size_inv])

        if len(treePos) > 0:
            treePos = np.array(treePos)
            self.plotTrees.set_data(treePos[:, 0], treePos[:, 1])
        else:
            self.plotTrees.set_data([], [])

        if len(firePos) > 0:
            firePos = np.array(firePos)
            self.plotFires.set_data(firePos[:, 0], firePos[:, 1])
            self.writer.grab_frame()
            self.writer.grab_frame()
        else:
            self.plotFires.set_data([], [])

        self.writer.grab_frame()