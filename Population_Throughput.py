# Human Population Throughput Template
# Births ➝ Living population ➝ Deaths

import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Define historical, current, and projected population data
population_data = {
    "Year": [-10000, -5000, 0, 1000, 1500, 1800, 1900, 1950, 2000, 2025, 2050, 2100, 2200, 2300, 2500],
    "Population_Billions": [0.001, 0.005, 0.3, 0.31, 0.5, 1.0, 1.65, 2.52, 6.14, 8.0, 9.7, 10.4, 10.0, 9.0, 8.0]
}

# Step 2: Create DataFrame
df = pd.DataFrame(population_data)

# Step 3: Estimate cumulative human lives (approximate method using trapezoidal integration)
# Assume linear population change between each period and average lifespan of ~70 years
df["Year_Interval"] = df["Year"].diff().fillna(0)
df["Average_Population"] = (df["Population_Billions"] + df["Population_Billions"].shift(1)) / 2
df["Average_Population"].fillna(0, inplace=True)
df["Interval_Throughput_Billions"] = df["Year_Interval"] * df["Average_Population"] / 70  # crude estimate

# Step 4: Calculate total human lives (throughput)
total_throughput = df["Interval_Throughput_Billions"].sum()

# Step 5: Plot population over time
plt.figure(figsize=(12, 6))
plt.plot(df["Year"], df["Population_Billions"], marker="o", linestyle="-")
plt.title("Global Human Population Over Time")
plt.xlabel("Year")
plt.ylabel("Population (Billions)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 6: Print total throughput
print(f"Estimated total human population throughput (to 2500): {total_throughput:.1f} billion humans")
