import matplotlib.pyplot as plt
import numpy as np

# Data points
x = np.array(range(1, 14))
y = np.array([67.83, 68.37, 69.67, 70.46, 71.58, 72.54, 72.62, 
              73.22, 74.60, 75.04, 75.42, 75.92, 76.22])

# Fit a line to the data
slope, intercept = np.polyfit(x, y, 1)  # Linear fit
y_fit = slope * x + intercept  # Compute the line of best fit

# Plotting the data points and the fitted line
plt.figure(figsize=(5, 5))
plt.plot(x, y, color='red')  # Data points
#plt.plot(x, y_fit, linestyle='--', color='black', label='Line of Best Fit')  # Fitted line

# Adding labels and title
plt.xlabel('Index', fontsize=22)
plt.ylabel('Accuracy (%)', fontsize=22)
plt.title('Percentage Values Over Time', fontsize=24)

# Adding a legend and grid
plt.legend()
plt.grid(True)

# Save the plot locally
#plt.savefig('percentage_values_fit_plot.png')

# Display the plot
plt.show()
