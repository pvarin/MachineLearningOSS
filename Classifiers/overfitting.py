import numpy as np
import matplotlib.pyplot as plt

# generate data along a quadratic
x = np.linspace(-1.5,1.5,99)
x_data = x[::4]
y_data = 2*x_data**2 + .25*np.random.randn(*x_data.shape)

# fit the data
coeff = np.polyfit(x_data,y_data,4)
y_fit = sum([c*x**i for i,c in enumerate(coeff[::-1])])
coeff = np.polyfit(x_data,y_data,len(x_data)-1)
y_ovrfit = sum([c*x**i for i,c in enumerate(coeff[::-1])])
print y_fit

# plot and label figures
plt.plot(x_data,y_data,'.')
plt.title('Well-Tuned Model')
plt.plot(x,y_fit)
plt.axis([-1.1,1.1,-.5,3])

plt.figure()
plt.plot(x_data,y_data,'.')
plt.title('Overfit Model')
plt.plot(x,y_ovrfit)
plt.axis([-1.1,1.1,-.5,3])

plt.show()