import numpy as np
import matplotlib.pyplot as plt

c = np.loadtxt('./samples.txt')
plt.plot(c[:,-2])
plt.show()
