import numpy as np
# import matplotlib.pyplot as plt

if __name__ == '__main__':
	x = np.linspace(-4,4)
	x_data = x[::5]
	y_data = x_data**2 + np.random.randn(*x_data.shape)

	coeff = np.polyfit(x_data,y_data,2)
	y_fit = sum([x*c**i for i,c in enumerate(coeff[::-1])])
	coeff = np.polyfit(x_data,y_data,len(x_data)-1)
	y_ovrfit = sum([x*c**i for i,c in enumerate(coeff[::-1])])
	print y_ovrfit

	# plt.plot(x,y)
	# plt.ylabel('some numbers')
	# plt.show()