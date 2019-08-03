from PIL import Image as Im #Работа с графическими изображениями
import tkinter #Работа с GUI
import numpy as np #Работа с массивами
import matplotlib
import matplotlib.pyplot as plt #Вывод на экран
import random


root = tkinter.Tk()
width = root.winfo_screenwidth()
height = root.winfo_screenheight()
pic_size = min(int(width/2), int(height/2))
print ("Pic size: (%s,%s)" % (pic_size, pic_size))

def show_image(image):
    # plt.imshow(image, cmap="gray")
    plt.imshow(image, cmap="gray", interpolation="none")
    plt.show()

def build_catmul_rom(vertexes, steps):
    pass

def build_ermit_spline(image, vertexes, derivatives, steps):
    for i in range(len(vertexes) - 1):
        x1 = vertexes[i, 0]
        x2 = vertexes[i + 1, 0]
        y1 = vertexes[i, 1]
        y2 = vertexes[i + 1, 1]
        f1 = derivatives[i]
        f2 = derivatives[i + 1]
        A = np.array([[x1**3, x1**2, x1, 1], [x2**3, x2**2, x2, 1], [3*(x1**2), 2*x1, 1, 0], [3*(x2**2), 2*x2, 1, 0]])
        b = np.array([y1, y2, f1, f2])
        x = np.linalg.inv(A).dot(b)
        d = (x2 - x1) / steps
        temp1 = x1
        temp2 = x1 + d
        temp_y1 = x[0]*(temp1**3) + x[1]*(temp1**2) + x[2]*temp1 + x[3]
        temp_y2 = x[0] * (temp2 ** 3) + x[1] * (temp2 ** 2) + x[2] * temp2 + x[3]
        while temp1 < x2:
            draw_line_bad_float(image, int(temp1*pic_size), int(temp_y1*pic_size), int(temp2*pic_size), int(temp_y2*pic_size), [255,255,255])
            temp1 = temp2
            temp2 = temp1 + d
            temp_y1 = x[0] * (temp1 ** 3) + x[1] * (temp1 ** 2) + x[2] * temp1 + x[3]
            temp_y2 = x[0] * (temp2 ** 3) + x[1] * (temp2 ** 2) + x[2] * temp2 + x[3]

def prepare_image():
    img = np.zeros(shape=(pic_size+1,pic_size+1, 3)).astype(np.uint8)
    return img

def vertexes_renderer(img, vertexes, color):
    for vertex in vertexes:
        img[tuple(vertex[:2])]=color
    return img

def draw_line_bad_float(img, x0, y0, x1, y1, color):
    steep = False
    #если ширина меньше высоты
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        steep = True
    #если первая координата больше второй
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    for x in range(x0, x1+1):
        if x1==x0:
            t = 0
        else:
            t = (x-x0) / (x1-x0)
        y = int(round(y0 * (1.-t) + y1 * t))
        #поменяли коорды, при отрисовке меняем обратно
        if (steep):
            img[x, y]=color
        else:
            img[y, x] = color

if __name__=="__main__":
    print(width, height)
