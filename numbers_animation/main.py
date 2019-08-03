from core import *
import time
import matplotlib
from matplotlib.animation import FuncAnimation, ArtistAnimation
import numpy as np

image=prepare_image()


vertexes_catmul=np.array([[0.0,0.3],[0.1,0.1],[0.5,0.2], [0.7,0.4], [0.9,0.0], [1.,1.]])

derivatives=np.array([0,1,2, -1])
vertexes_ermit=np.array([[0.1,0.1],[0.5,0.2], [0.7,0.4], [0.9,0.0]])

build_catmul_rom(vertexes_catmul, 200)

build_ermit_spline(image, vertexes_ermit, derivatives, 200)
image = np.rot90(image, 2, axes=(0, 1))
show_image(image)