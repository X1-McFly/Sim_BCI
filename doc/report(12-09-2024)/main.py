import matplotlib.pyplot as plt
import numpy as np

# Financial estimates (example values)
years = np.arange(2018, 2025)  # Years from 2023 to 2032
historic_market_size = [1.0, 1.2, 1.3, 1.5, 1.7, 2.0, 2.3]

grandview_prediction_years = np.arange(2024,2030)
grandview_prediction = [2.3, 2.7, 3.1, 3.6, 4.4, 5.2]

mordor_prediction_years = [2024, 2029]
mordor_prediction = [2.3, 3.25]

bcc_prediction_years = [2024, 2029]
bcc_prediction = [2.3, 4.5]

# Plot
plt.plot(years, historic_market_size, marker='o', label="BCI Market Valuation", color="black")
plt.plot(grandview_prediction_years, grandview_prediction, marker='o', label="Grandview Research", color="blue")
plt.plot(mordor_prediction_years, mordor_prediction, marker='o', label="Mordor Intelligence", color="red")
plt.plot(bcc_prediction_years, bcc_prediction, marker='o', label="BCC Research", color="green")
plt.axvline(x=2024, linestyle='--', color='gray')
plt.title("Estimated Growth of the BCI Market (2024-2030)", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Market Size (Billion USD)", fontsize=12)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()