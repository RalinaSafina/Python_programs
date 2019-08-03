import numpy as np
import re
def read_vertexes(path_to_file, x_axis, y_axis, z_axis):
    with open(path_to_file, 'r', encoding='ansi') as f_txt:
        lines = filter(None, (line.rstrip() for line in f_txt))
        for i in lines:
            line = i.split()
            if line[0] == 'v':
                x_axis.append(float(line[1]))
                y_axis.append(float(line[2]))
                z_axis.append(float(line[3]))

def read_frames(path_to_file, f1, f2, f3, t1, t2, t3, n1, n2, n3):
    with open(path_to_file, 'r', encoding='ansi') as f_txt:
        lines = filter(None, (line.rstrip() for line in f_txt))
        for i in lines:
            line = re.sub('[/]', ' ', i)
            line = line.split()
            if line[0] == 'f':
                f1.append(int(line[1]) - 1)
                f2.append(int(line[4]) - 1)
                f3.append(int(line[7]) - 1)
                t1.append(int(line[2]) - 1)
                t2.append(int(line[5]) - 1)
                t3.append(int(line[8]) - 1)
                n1.append(int(line[3]) - 1)
                n2.append(int(line[6]) - 1)
                n3.append(int(line[9]) - 1)

def read_texture(path_to_file, u, v):
    with open(path_to_file, 'r', encoding='ansi') as f_txt:
        lines = filter(None, (line.rstrip() for line in f_txt))
        for i in lines:
            line = i.split()
            if line[0] == 'vt':
                u.append(float(line[1]))
                v.append(float(line[2]))

def read_normal(path_to_file, n_x, n_y, n_z):
    with open(path_to_file, 'r', encoding='ansi') as f_txt:
        lines = filter(None, (line.rstrip() for line in f_txt))
        for i in lines:
            line = i.split()
            if line[0] == 'vn':
                n_x.append(float(line[1]))
                n_y.append(float(line[2]))
                n_z.append(float(line[3]))
