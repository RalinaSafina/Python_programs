import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()

image = np.eye(512)
im = plt.imshow(image, animated=True)

x,y=512,512

def updatefig(args):
    global x, y
    im.set_array(np.random.rand(x,y))
    return im,

ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
plt.show()