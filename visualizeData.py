import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the training data
df_train = pd.read_csv('train.csv')

# Set up the plotting style
sns.set(style="whitegrid")

# Create a figure for the plots
plt.figure(figsize=(14, 6))

# Plot 1: Histogram of HotelValue
plt.subplot(1, 2, 1)
sns.histplot(df_train['HotelValue'], kde=True)
plt.title('Distribution of HotelValue')
plt.xlabel('Hotel Value')
plt.ylabel('Frequency')

# Plot 2: Q-Q plot to check for normality
from scipy import stats
plt.subplot(1, 2, 2)
stats.probplot(df_train['HotelValue'], plot=plt)
plt.title('Q-Q Plot of HotelValue')

plt.tight_layout()
plt.show()

# Print skewness
print(f"Skewness of HotelValue: {df_train['HotelValue'].skew():.2f}")
# Skewness is 1.73

# Calculate correlations
# We must use 'numeric_only=True' to ignore categorical columns
correlations = df_train.corr(numeric_only=True)

# Get the top 10 features most correlated with HotelValue
top_10_corr = correlations['HotelValue'].abs().sort_values(ascending=False).head(11)[1:] # [1:] to skip HotelValue itself

print("Top 10 Most Correlated Numerical Features with HotelValue:")
print(top_10_corr)

# Plot a heatmap of these top 10 features
top_10_features = top_10_corr.index.tolist() + ['HotelValue']
plt.figure(figsize=(12, 8))
sns.heatmap(df_train[top_10_features].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Top 10 Features')
plt.show()

# Create a boxplot for OverallQuality vs. HotelValue
plt.figure(figsize=(12, 7))
sns.boxplot(x='OverallQuality', y='HotelValue', data=df_train)
plt.title('HotelValue vs. Overall Quality')
plt.xlabel('Overall Quality')
plt.ylabel('Hotel Value')
plt.show()