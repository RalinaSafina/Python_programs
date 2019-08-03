import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import copy
import math

class Digits:

    digits=\
    np.array([
        # zero
        [[159.0, 84.0, 123.0, 158.0, 131.0, 258.0],
        [139.0, 358.0, 167.0, 445.0, 256.0, 446.0],
        [345.0, 447.0, 369.0, 349.0, 369.0, 275.0],
        [369.0, 201.0, 365.0, 81.0, 231.0, 75.0]],
        # one
        [[226.0, 99.0, 230.0, 58.0, 243.0, 43.0],
        [256.0, 28.0, 252.0, 100.0, 253.0, 167.0],
        [254.0, 234.0, 254.0, 194.0, 255.0, 303.0],
        [256.0, 412.0, 254.0, 361.0, 255.0, 424.0]],
        # two
        [[152.0, 55.0, 208.0, 26.0, 271.0, 50.0],
        [334.0, 74.0, 360.0, 159.0, 336.0, 241.0],
        [312.0, 323.0, 136.0, 454.0, 120.0, 405.0],
        [104.0, 356.0, 327.0, 393.0, 373.0, 414.0]],
        # three
        [[113.0, 14.0, 267.0, 17.0, 311.0, 107.0],
        [355.0, 197.0, 190.0, 285.0, 182.0, 250.0],
        [174.0, 215.0, 396.0, 273.0, 338.0, 388.0],
        [280.0, 503.0, 110.0, 445.0, 93.0, 391.0]],
        # four
        [[249.0, 230.0, 192.0, 234.0, 131.0, 239.0],
        [70.0, 244.0, 142.0, 138.0, 192.0, 84.0],
        [242.0, 30.0, 283.0, -30.0, 260.0, 108.0],
        [237.0, 246.0, 246.0, 435.0, 247.0, 438.0]],
        # five
        [[226.0, 42.0, 153.0, 44.0, 144.0, 61.0],
        [135.0, 78.0, 145.0, 203.0, 152.0, 223.0],
        [159.0, 243.0, 351.0, 165.0, 361.0, 302.0],
        [371.0, 439.0, 262.0, 452.0, 147.0, 409.0]],
        # six
        [[191.0, 104.0, 160.0, 224.0, 149.0, 296.0],
        [138.0, 368.0, 163.0, 451.0, 242.0, 458.0],
        [321.0, 465.0, 367.0, 402.0, 348.0, 321.0],
        [329.0, 240.0, 220.0, 243.0, 168.0, 285.0]],
        # seven
        [[168.0, 34.0, 245.0, 42.0, 312.0, 38.0],
        [379.0, 34.0, 305.0, 145.0, 294.0, 166.0],
        [283.0, 187.0, 243.0, 267.0, 231.0, 295.0],
        [219.0, 323.0, 200.0, 388.0, 198.0, 452.0]],
        # eight
        [[336.0, 184.0, 353.0, 52.0, 240.0, 43.0],
        [127.0, 34.0, 143.0, 215.0, 225.0, 247.0],
        [307.0, 279.0, 403.0, 427.0, 248.0, 432.0],
        [93.0, 437.0, 124.0, 304.0, 217.0, 255.0]],
        # nine
        [[323.0, 6.0, 171.0, 33.0, 151.0, 85.0],
        [131.0, 137.0, 161.0, 184.0, 219.0, 190.0],
        [277.0, 196.0, 346.0, 149.0, 322.0, 122.0],
        [298.0, 95.0, 297.0, 365.0, 297.0, 448.0]]

    ])
    end_points=\
    np.array([
        # zero
        [254, 47],
        # one
        [138, 180],
        # two
        [104, 111],
        # three
        [96, 132],
        # four
        [374, 244],
        # five
        [340, 52],
        # six
        [301, 26],
        # seven
        [108, 52],
        # eight
        [243,242],
        # nine
        [322, 105]
    ]).astype(float)
    def __init__(self):
        pass
    def get_coords(self, digit):
        return self.digits[digit]

class Clock:

    def __init__(self, frames_to_change=60):
        self.frames_to_change=frames_to_change
        self.finish=False
        self.Digits=Digits()
        self.fig, self.ax = plt.subplots()
        self.fig.suptitle("Clock")
        self.img = np.zeros((512,512,3), dtype=np.uint8)
        self.blank = np.zeros((512,512,3), dtype=np.uint8)
        cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.im = plt.imshow(self.img, animated=True)
        self.ani = FuncAnimation(self.fig, self.plot_digit, init_func=self.init_anim, frames=self.end_clock, blit=True, interval=5)
        plt.show()
        pass

    def end_clock(self):
        ii = 0
        while not self.finish:
            ii += 1
            yield ii

    def init_anim(self, digit=0):
        #текущая цифра
        self.img.fill(0)
        self.cur_digit = digit
        #текущий фрейм в превращении
        self.cur_step = 0
        #текущие точки
        self.current_besier_points = self.Digits.get_coords(self.cur_digit).copy()
        self.current_end_point = self.Digits.end_points[self.cur_digit].copy()
        #текущие сдвиги по точкам
        self.current_shift_per_step_vertexes=(self.Digits.digits[self.cur_digit+1 if self.cur_digit<9 else 0]-self.Digits.digits[self.cur_digit])/self.frames_to_change
        self.current_shift_per_step_end_point=(self.Digits.end_points[self.cur_digit+1 if self.cur_digit<9 else 0]-self.Digits.end_points[self.cur_digit])/self.frames_to_change

        return self.im,

    def onclick(self, event):
        self.finish=True

    def casteljau(self, ar_x, ar_y, acc):
        #Запуск рекурсивного алгоритма Кастельжо, acc - шаг по t
        x = []
        y = []
        for i in np.arange(0, 1, acc):
            coord = self.casteljau_rec(ar_x, ar_y, i)
            x = np.append(x, coord[0])
            y = np.append(y, coord[1])
        return x, y

    def casteljau_rec(self, ar_x, ar_y, t):
        if np.size(ar_x) == 1:
            return ar_x, ar_y
        x = []
        y = []
        for i in range(len(ar_x) - 1):
            x = np.append(x, ar_x[i + 1]*t + ar_x[i]*(1 - t))
            y = np.append(y, ar_y[i + 1] * t + ar_y[i] * (1 - t))
        return self.casteljau_rec(x, y, t)

    def draw_line(self, x0, y0, x1, y1):
        steep = False
        if math.fabs(x0 - x1) < math.fabs(y0 - y1):
            steep = True
            x0, y0 = y0, x0
            x1, y1 = y1, x1
        if x1 < x0:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        t = math.fabs(y1 - y0)
        error = 0
        y = y0
        for x in range(x0, x1 + 1):
            if steep:
                self.img[y, x] = (255, 255, 255)
            else:
                self.img[x, y] = (255, 255, 255)
            error += t
            if error * 2 >= math.fabs(x1 - x0):
                y = y + np.sign(y1 - y0) * 1
                error -= math.fabs(x1 - x0)


    def plot_digit(self, par):
        # Если нарисовали все фреймы - инициализация новой цифры
        # Иначе - запустить построение кривой безье на основе данных current_besier_points и current_end_point
        # с заданной точностью acc. Связать все полученные точки последовательно через брезенхема и нарисовать.
        # Произвести сдвиг current_besier_points и current_end_point на shift

        x = []
        y = []
        for j in range(4):
            ar_x = []
            ar_y = []
            temp_x = 5
            temp_y = 4
            for i in range(3):
                ar_x = np.append(ar_x, self.current_besier_points[3 - j, temp_x])
                ar_y = np.append(ar_y, self.current_besier_points[3 - j, temp_y])
                temp_x -= 2
                temp_y -= 2
            if j == 3:
                ar_x = np.append(ar_x, self.current_end_point[1])
                ar_y = np.append(ar_y, self.current_end_point[0])
            else:
                ar_x = np.append(ar_x, self.current_besier_points[2 - j, 5])
                ar_y = np.append(ar_y, self.current_besier_points[2 - j, 4])
            x, y = self.casteljau(ar_x, ar_y, 0.1)
            for i in range(len(x) - 1):
                self.draw_line(int(x[i]), int(y[i]), int(x[i + 1]), int(y[i + 1]))
        self.im.set_array(self.img)
        self.current_besier_points += self.current_shift_per_step_vertexes
        self.current_end_point += self.current_shift_per_step_end_point
        self.cur_step += 1
        self.img.fill(0)
        if self.cur_step == self.frames_to_change:
            digit = self.cur_digit
            if digit == 9:
                self.init_anim()
            else:
                self.init_anim(digit + 1)
        return self.im,




clock=Clock()
#clock.plot_digit(0)


