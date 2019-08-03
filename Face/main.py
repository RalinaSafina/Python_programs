import core
import vizualization
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as pi

x_axis = ([])
y_axis = ([])
z_axis = ([])
f1 = ([])
f2 = ([])
f3 = ([])
t1 = ([])
t2 = ([])
t3 = ([])
n1 = ([])
n2 = ([])
n3 = ([])
u = ([])
v = ([])
n_x = ([])
n_y = ([])
n_z = ([])
light = ([0, 0, -1])
img_size = 512
r = 256
l = -256
t = 256
b = -256
n = 70
f = 500 + n
z_buff = vizualization.z_buf(img_size)

im = pi.open('C:/Face/african_head_diffuse.tga')
img_t = np.rot90(np.array(im), 3, axes=(0, 1))
core.read_vertexes('C:/Face/face.obj', x_axis, y_axis, z_axis)
core.read_frames('C:/Face/face.obj', f1, f2, f3, t1, t2, t3, n1, n2, n3)
core.read_texture('C:/Face/face.obj', u, v)
core.read_normal('C:/Face/face.obj', n_x, n_y, n_z)
vizualization.draw_vertexes(x_axis, y_axis, z_axis, img_size)
img = np.zeros((512, 512, 3), dtype=np.uint8)
#vizualization.rotate_light(degree = -45, light = light)
vizualization.rotate(x_axis, y_axis, z_axis, n_x, n_y, n_z, degree = 180)
#vizualization.zoom_shift(x_axis, y_axis, z_axis, 500, max(z_axis)*500 + (f + n)/2)
vizualization.projective(x_axis, y_axis, z_axis, r, l, t, b, f, n)
vizualization.draw_image(x_axis, y_axis, z_axis, f1, f2, f3, t1, t2, t3, n1, n2, n3, img, img_size, light, z_buff, u, v, img_t, n_x, n_y, n_z)
img = np.rot90(img, 1, axes=(0, 1))
plt.imshow(img)
plt.show()
