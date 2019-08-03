import numpy as np
import matplotlib.pyplot as plt

def prediction(x, t, M):
    PM = np.empty((len(x), M))
    for i in range(0, len(x)):
        for j in range(0, M):
            PM[i, j] = x[i] ** j
    w = np.linalg.inv(PM.transpose() @ PM) @ PM.transpose() @ t.transpose()
    y = w.transpose() @ PM.transpose()
    return y

x = np.linspace(0, 1, 1000)
z = 20*np.sin(2*np.pi*3*x) + 100*np.exp(x)
e = 10*np.random.randn(1000)
t = z + e
E = []
help = []
for i in range(100):
    y = prediction(x, t, i)
    temp = 0
    for j in range(len(x)):
        temp += (y[j] - t[j])**2
    E.append(temp/2)
    help.append(i)

y1 = prediction(x, t, 1)
y2 = prediction(x, t, 8)
y3 = prediction(x, t, 100)
plt.figure()
plt.subplot(2, 2, 1)
plt.plot(x, t, 'r,')
plt.plot(x, z, '-g')
plt.plot(x, y1, '-b')
plt.title("M = 1")
plt.subplot(2, 2, 2)
plt.plot(x, t, 'r,')
plt.plot(x, z, '-g')
plt.plot(x, y2, '-b')
plt.title("M = 8")
plt.subplot(2, 2, 3)
plt.plot(x, t, 'r,')
plt.plot(x, z, '-g')
plt.plot(x, y3, '-b')
plt.title("M = 100")
plt.subplot(2, 2, 4)
plt.plot(help, E, '-')
plt.title("Зависимость Е от М")

plt.show()

