import numpy as np
import matplotlib.pyplot as plt

fig, _ = plt.subplots()
one_tick = fig.axes[0].yaxis.get_major_ticks()[0]
plt.plot()


matrix = np.random.random(size=100).reshape(10,10)
print(matrix)