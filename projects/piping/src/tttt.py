import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def clog(x):
    x = -np.exp(x)
    x = np.exp(x)
    return 1-x 

x = np.linspace(-10, 10, 1000)
y = clog(x)
plt.plot(x, y)
plt.show()



