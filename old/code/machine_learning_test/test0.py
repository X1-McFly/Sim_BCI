from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import random
import math

x = []
y = []
y_list = []
y_average = []
limit = 1000
model = LinearRegression()

for i in range(limit):
    plt.clf()
    
    x.append(i)
    y.append(math.sin(x[i]) * (2 * math.cos(x[i]+random.randint(1,5))+1))

    y_list.append(y[i])

    if len(y_list) > 50:
        y_list.pop(0)

    if len(y_list) == 50:
        y_average.append(sum(y_list) / len(y_list))        
    else:
        y_average.append(0)

    x_sample = np.array(x).reshape(-1, 1)
    y_sample = np.array(y).reshape(-1, 1)

    model.fit(x_sample, y_sample)

    result = model.predict(x_sample)
    x_np = np.array(x).reshape(-1, 1)
    
    plt.xlim(0, limit)
    plt.ylim(-5, 5)
    plt.ylabel("Voltage (V)")
    plt.xlabel("Time (S)")
    plt.plot(x, y, color="black")
    plt.plot(x, y_average, color="red")
    plt.plot(x_np, result, color="blue")
    plt.pause(0.00001)

plt.ioff()
plt.show()