import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import core
import vizualization

class Head:

    def __init__(self, pic_size = 512):
        self.pic_size = pic_size
        self.finish = False
        self.fig, self.ax = plt.subplots()
        self.img = np.zeros((pic_size, pic_size, 3), dtype=np.uint8)
        self.im = plt.imshow(self.img, animated=True)
        self.ani = FuncAnimation(self.fig, self.update, init_func=self.init_anim, frames=self.end, blit=True, interval=5)
        plt.show()


    def end(self):
        i = 0
        while not self.finish:
            i += 1
            yield i

    def init_anim(self, degree = 0):
        self.img = np.zeros((self.pic_size, self.pic_size, 3), dtype=np.uint8)
        self.degree = degree
        return self.im,

    def update(self, par):
        x_axis = ([])
        y_axis = ([])
        z_axis = ([])
        f1 = ([])
        f2 = ([])
        f3 = ([])
        light = ([0, 0, 1])
        self.img = np.zeros((512, 512, 3), dtype=np.uint8)
        core.read_vertexes('C:/Face/face.obj', x_axis, y_axis, z_axis)
        vizualization.draw_vertexes(x_axis, y_axis, z_axis, self.pic_size)
        core.read_frames('C:/Face/face.obj', f1, f2, f3)
        vizualization.rotate(x_axis, y_axis, z_axis, self.degree)
        #vizualization.rotate_light(self.degree, light)         #Раскомментить, чтобы свет фиксировался, т.е. постоянно освещалась лишь определенная часть лица
        self.degree = self.degree + 15
        vizualization.draw_image(x_axis, y_axis, z_axis, f1, f2, f3, self.img, self.pic_size, light)
        self.im.set_array(np.rot90(self.img))
        return self.im,

gg = Head()
