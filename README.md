# PLP-ACADEMY-WEEK-7-ASSIGNMENT
WEEK 7 ASSIGNMENT

# Task 1: Load and Explore the Dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: Set seaborn style
sns.set(style="whitegrid")

# Error handling while loading the dataset
try:
    # Load the dataset (e.g., iris dataset from seaborn)
    df = sns.load_dataset('iris')
    print("Dataset loaded successfully.\n")
except Exception as e:
    print(f"Error loading dataset: {e}")

# Display first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Data structure
print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

# Clean missing values (if any)
df = df.dropna()

# Task 2: Basic Data Analysis
print("\nDescriptive Statistics:")
print(df.describe())

# Grouping by species and computing average petal length
print("\nAverage Petal Length by Species:")
print(df.groupby('species')['petal_length'].mean())

# Task 3: Visualizations

# 1. Line Chart: Showing sepal_length over index per species
plt.figure(figsize=(10, 5))
for species in df['species'].unique():
    species_data = df[df['species'] == species]
    plt.plot(species_data.index, species_data['sepal_length'], label=species)
plt.title('Sepal Length Trend by Species')
plt.xlabel('Index')
plt.ylabel('Sepal Length')
plt.legend()
plt.tight_layout()
plt.show()

# 2. Bar Chart: Average petal length per species
plt.figure(figsize=(7, 5))
df.groupby('species')['petal_length'].mean().plot(kind='bar', color='skyblue')
plt.title('Average Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length')
plt.tight_layout()
plt.show()

# 3. Histogram: Distribution of sepal length
plt.figure(figsize=(7, 5))
plt.hist(df['sepal_length'], bins=15, color='orange', edgecolor='black')
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 4. Scatter Plot: Sepal Length vs Petal Length
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x='sepal_length', y='petal_length', hue='species')
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.tight_layout()
plt.show()

# Optional: Observations
print("\nFindings:")
print("- Setosa has shorter petals and sepals compared to other species.")
print("- Sepal length and petal length have a positive correlation.")
print("- Virginica shows the highest average petal length.")

