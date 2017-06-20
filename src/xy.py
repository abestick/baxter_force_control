import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 10, 0.01)
y1 = -x**2 + 9*x + 11
y2 = x**2 - 12*x + 80

plt.plot(x, y1)
plt.xlabel('x')
plt.ylabel('y')
plt.title(r'$y = 11 + 9x - x^2$')
plt.show()
plt.plot(x, y2)
plt.xlabel('x')
plt.ylabel('y')
plt.title(r'$y = 80 - 12x + x^2$')
plt.show()