import matplotlib.pyplot as plt
from helperfun import deepNetFun
import numpy as np

fun = deepNetFun()
x = np.linspace(-3, 3, 1000)
y = fun.de_elu(x)
plt.plot(x, y)
plt.title("derivative ELU")
#plt.title("derivative Leaky ReLU")
plt.grid()
plt.show()
