import numpy as np
import matplotlib.pyplot as plt

plt.close("all")
# Compute the x and y coordinates for points on a sine curve
x = np.arange(-2*np.pi, 3 * np.pi, 0.1)
y=[]
for i in range(0,10):
    y.append(np.tanh(x+i-5))

# Plot the points using matplotlib
for i in range(len(y)):
    plt.plot(x, y[i])
    plt.legend(str(i))
plt.title("this is a test")
plt.plot(x,np.sin(x),label="sine")
plt.show()  # You must call plt.show() to make graphics appear.