import matplotlib as mpl

mpl.use('TkAgg')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Sample data
#data = sns.load_dataset("tips")  # Load a built-in dataset as an example

data = np.array([1,3,5])
# Create a box plot
sns.boxplot(data=data, showfliers=False)
print(data)

# Add labels and title
plt.xlabel("Day of the week")
plt.ylabel("Total Bill Amount ($)")
plt.title("Box Plot of Total Bill Amount by Day")

# Show the plot
plt.show()
raise


# Create a DataFrame from your CSV data
#data = pd.read_csv('exports/summary_table.csv')

array1= [1,3,5]
array2=[2,7,4]
# Create a DataFrame from two arrays of variables (replace with your own data)
data = pd.DataFrame({'Variable1': array1, 'Variable2': array2})

# Calculate the difference between the two variables
data['Difference'] = data['Variable1'] - data['Variable2']

# Create a violin plot
sns.violinplot(x='Difference', data=data)

# Print the standard deviation of the difference
std_deviation = data['Difference'].std()
print(f"Standard Deviation of Difference: {std_deviation}")

# Customize the plot
plt.title("Violin Plot of Difference")
plt.xlabel("Difference")
plt.ylabel("Density")
plt.show()