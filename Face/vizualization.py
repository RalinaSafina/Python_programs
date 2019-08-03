import math
import numpy as np

def draw_vertexes(x_axis, y_axis, z_axis, img_size):
    x = max(x_axis) - min(x_axis)
    y = max(y_axis) - min(y_axis)
    z = max(z_axis) - min(z_axis)
    koef = math.floor((img_size-1) / max(x, y, z))
    for i in range(len(x_axis)):
        x_axis[i] = int(x_axis[i] * koef)
        y_axis[i] = int(y_axis[i] * koef)
        z_axis[i] = int(z_axis[i] * koef + 245)
    '''if min(x_axis) < 0:
        x_shift = math.fabs(min(x_axis) * koef)
    elif min(x_axis) > 0:
        x_shift = -math.fabs(min(x_axis) * koef)
    else:
        x_shift = 0
    if min(y_axis) < 0:
        y_shift = math.fabs(min(y_axis) * koef)
    elif min(y_axis) > 0:
        y_shift = -math.fabs(min(y_axis) * koef)
    else:
        y_shift = 0
    if min(z_axis) < 0:
        z_shift = math.fabs(min(z_axis) * koef)
    elif min(z_axis) > 0:
        z_shift = -math.fabs(min(z_axis) * koef)
    else:
        z_shift = 0
    for i in range(len(x_axis)):
        x_axis[i] = int(x_axis[i] * koef + x_shift)
        y_axis[i] = int(y_axis[i] * koef + y_shift)
        z_axis[i] = int(z_axis[i] * koef + z_shift)
    x = max(x_axis) - min(x_axis)
    y = max(y_axis) - min(y_axis)
    z = max(z_axis) - min(z_axis)
    if x < max(x, y, z):
        for i in range(len(x_axis)):
            x_axis[i] = int(x_axis[i] + (img_size-1)/2 - x/2)
    if y < max(x, y, z):
        for i in range(len(y_axis)):
            y_axis[i] = int(y_axis[i] + (img_size-1)/2 - y/2)
    if z < max(x, y, z):
        for i in range(len(z_axis)):
            z_axis[i] = int(z_axis[i] + (img_size-1)/2 - z/2)'''

def draw_line_good(x0, y0, z0,  x1, y1, z1, img, color, z_buff):
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
            img[y, x] = color
        else:
            img[x, y] = color
        error += t
        if error * 2 >= math.fabs(x1 - x0):
            y = y + np.sign(y1 - y0) * 1
            error -= math.fabs(x1 - x0)

def draw_image(x_axis, y_axis, z_axis, f1, f2, f3, t1, t2, t3, n1, n2, n3, img, img_size, light, z_buff, u, v, img_t, n_x, n_y, n_z):
    for i in range(len(f1)):
        p = ([x_axis[f2[i]] - x_axis[f1[i]], y_axis[f2[i]] - y_axis[f1[i]], z_axis[f2[i]] - z_axis[f1[i]]])
        q = ([x_axis[f3[i]] - x_axis[f1[i]], y_axis[f3[i]] - y_axis[f1[i]], z_axis[f3[i]] - z_axis[f1[i]]])
        N = np.cross(p, q)
        lengh_N = math.sqrt(N[0] * N[0] + N[1] * N[1] + N[2] * N[2])
        lengh_L = math.sqrt(light[0] * light[0] + light[1] * light[1] + light[2] * light[2])
        cos = light[0]/lengh_L * N[0]/lengh_N + light[1]/lengh_L * N[1]/lengh_N + light[2]/lengh_L * N[2]/lengh_N
        if cos > 0:
            for q in range(min([x_axis[f1[i]], x_axis[f2[i]], x_axis[f3[i]]]),
                           max([x_axis[f1[i]], x_axis[f2[i]], x_axis[f3[i]]])):
                for j in range(min([y_axis[f1[i]], y_axis[f2[i]], y_axis[f3[i]]]),
                               max([y_axis[f1[i]], y_axis[f2[i]], y_axis[f3[i]]])):
                    if q < img_size and j < img_size and q > 0 and j > 0:
                        '''a = (x_axis[f1[i]] - q) * (y_axis[f2[i]] - y_axis[f1[i]]) - \
                            (x_axis[f2[i]] - x_axis[f1[i]]) * (y_axis[f1[i]] - j)
                        b = (x_axis[f2[i]] - q) * (y_axis[f3[i]] - y_axis[f2[i]]) - \
                            (x_axis[f3[i]] - x_axis[f2[i]]) * (y_axis[f2[i]] - j)
                        c = (x_axis[f3[i]] - q) * (y_axis[f1[i]] - y_axis[f3[i]]) - \
                            (x_axis[f1[i]] - x_axis[f3[i]]) * (y_axis[f3[i]] - j) ''' #если раскомментить, то изменить буфер
                        c = ((j - y_axis[f1[i]]) * (x_axis[f2[i]] - x_axis[f1[i]]) - \
                             (q - x_axis[f1[i]]) * (y_axis[f2[i]] - y_axis[f1[i]])) / \
                            ((y_axis[f3[i]] - y_axis[f1[i]]) * (x_axis[f2[i]] - x_axis[f1[i]]) - \
                            (x_axis[f3[i]] - x_axis[f1[i]]) * (y_axis[f2[i]] - y_axis[f1[i]]))
                        b = ((j - y_axis[f1[i]]) * (x_axis[f3[i]] - x_axis[f1[i]]) - \
                             (q - x_axis[f1[i]]) * (y_axis[f3[i]] - y_axis[f1[i]])) / \
                            ((y_axis[f2[i]] - y_axis[f1[i]]) * (x_axis[f3[i]] - x_axis[f1[i]]) - \
                            (x_axis[f2[i]] - x_axis[f1[i]]) * (y_axis[f3[i]] - y_axis[f1[i]]))
                        a = ((j - y_axis[f3[i]]) * (x_axis[f2[i]] - x_axis[f3[i]]) - \
                             (q - x_axis[f3[i]]) * (y_axis[f2[i]] - y_axis[f3[i]])) / \
                            ((y_axis[f1[i]] - y_axis[f3[i]]) * (x_axis[f2[i]] - x_axis[f3[i]]) - \
                            (x_axis[f1[i]] - x_axis[f3[i]]) * (y_axis[f2[i]] - y_axis[f3[i]]))
                        z = 1 / (a / z_axis[f1[i]] + b / z_axis[f2[i]] + c / z_axis[f3[i]])
                        if ((a >= 0 and b >= 0 and c >= 0) or (a <= 0 and b <= 0 and c <= 0)) and z < z_buff[q, j]:
                            z_buff[q, j] = z
                            u1 = (a * u[t1[i]]/z_axis[f1[i]] + b * u[t2[i]]/z_axis[f2[i]] + c * u[t3[i]]/z_axis[f3[i]])*z
                            v1 = (a * v[t1[i]]/z_axis[f1[i]] + b * v[t2[i]]/z_axis[f2[i]] + c * v[t3[i]]/z_axis[f3[i]])*z
                            norm_x = a * n_x[n1[i]] + b * n_x[n2[i]] + c * n_x[n3[i]]
                            norm_y = a * n_y[n1[i]] + b * n_y[n2[i]] + c * n_y[n3[i]]
                            norm_z = a * n_z[n1[i]] + b * n_z[n2[i]] + c * n_z[n3[i]]
                            norm_len = math.sqrt(norm_x * norm_x + norm_y * norm_y + norm_z * norm_z)
                            cos_1 = light[0] / lengh_L * norm_x / norm_len + light[1] / lengh_L * norm_y / norm_len + light[2] / lengh_L * norm_z / norm_len
                            view_vec = ([0, 0, -1])
                            view_vec_len = math.sqrt(view_vec[0] * view_vec[0] + view_vec[1] * view_vec[1] + view_vec[2] * view_vec[2])
                            alfa = 30
                            normal = np.array([norm_x, norm_y, norm_z])
                            r = light - 2 * normal * np.dot(light, normal)
                            r_len = math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
                            cos_2 = np.dot(r, view_vec)/(r_len*view_vec_len)
                            if cos_2 > 0:
                                cos_2 = 0
                            if cos_1 > 0:
                                img[q, j] = (min(int(round(img_t[int(round(u1 * np.size(img_t, axis=0))),
                                                             int(round(v1 * np.size(img_t, axis=1))), 0] * cos_1 + 100*(cos_2**alfa))), 255),
                                            min(int(round(img_t[int(round(u1 * np.size(img_t, axis=0))),
                                                            int(round(v1 * np.size(img_t, axis=1))), 1] * cos_1 + 100*(cos_2**alfa))), 255),
                                            min(int(round(img_t[int(round(u1 * np.size(img_t, axis=0))),
                                                            int(round(v1 * np.size(img_t, axis=1))), 2] * cos_1 + 100*(cos_2**alfa))), 255))

def rotate(x_axis, y_axis, z_axis, n_x, n_y, n_z, degree):
    rot = np.array([[np.cos(np.radians(degree)), 0, -np.sin(np.radians(degree)), 0],
                    [0, 1, 0, 0],
                    [np.sin(np.radians(degree)), 0, np.cos(np.radians(degree)), 0],
                    [0, 0, 0, 1]])
    shift1 = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [-(max(x_axis) + min(x_axis)) / 2, -(max(y_axis) + min(y_axis)) / 2, -(max(z_axis) + min(z_axis)) / 2, 1]])
    shift2 = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [(max(x_axis) + min(x_axis)) / 2, (max(y_axis) + min(y_axis)) / 2, (max(z_axis) + min(z_axis)) / 2, 1]])
    for i in range(len(x_axis)):
        array = np.array([x_axis[i], y_axis[i], z_axis[i], 1])
        array = array.dot(shift1).dot(rot).dot(shift2)
        x_axis[i] = int(round(array[0]))
        y_axis[i] = int(round(array[1]))
        z_axis[i] = int(round(array[2]))
    for i in range(len(n_x)):
        array = np.array([n_x[i], n_y[i], n_z[i], 1])
        array = array.dot(np.linalg.inv(rot).transpose())
        n_x[i] = array[0]
        n_y[i] = array[1]
        n_z[i] = array[2]

def rotate_light(degree, light):
    rot = np.array([[np.cos(np.radians(degree)), 0, -np.sin(np.radians(degree)), 0],
                    [0, 1, 0, 0],
                    [np.sin(np.radians(degree)), 0, np.cos(np.radians(degree)), 0],
                    [0, 0, 0, 1]])
    l = np.array([light[0], light[1], light[2], 1])
    l = l.dot(rot)
    light[0] = int(round(l[0]))
    light[1] = int(round(l[1]))
    light[2] = int(round(l[2]))

def zoom_shift(x_axis, y_axis, z_axis, times, zz):
    zoom = np.array(([[times, 0, 0, 0],
                      [0, times, 0, 0],
                      [0, 0, times, 0],
                      [0, 0, 0, 1]]))
    for i in range(len(x_axis)):
        array = np.array([x_axis[i], y_axis[i], z_axis[i], 1])
        array = array.dot(zoom)
        x_axis[i] = int(round(array[0]))
        y_axis[i] = int(round(array[1]))
        z_axis[i] = int(round(array[2])) - zz

def projective(x_axis, y_axis, z_axis, r, l, t, b, f, n):
    proj = np.array([[2*n / (r - l), 0, (r + l) / (r - l), 0],
                     [0, 2*n / (t - b), (t + b) / (t - b), 0],
                     [0, 0, (f + n) / (f - n), -2 * f * n / (f - n)],
                     [0, 0, 1, 0]], dtype=np.float64).transpose()
    for i in range(len(x_axis)):
        array1 = np.array([x_axis[i], y_axis[i], z_axis[i], 1])
        array = array1.dot(proj)
        x_axis[i] = int(round((r - l) * array[0] / array[3] / 2 + (r - l) / 2))
        y_axis[i] = int(round((t - b) * array[1] / array[3] / 2 + (t - b) / 2))
        z_axis[i] = int(round((f - n) * array[2] / array[3] / 2 + (f - n) / 2))

def z_buf(img_size):
    z = np.arange(img_size**2).reshape(img_size, img_size)
    for i in range(img_size):
        for j in range(img_size):
            z[i, j] = 1000000
    return z
