
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('transaction_data.csv')

# Plot transaction amount distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['transaction_amount'], bins=50, kde=True)
plt.title('Transaction Amount Distribution')
plt.show()

# Plot fraud distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='is_fraud', data=data)
plt.title('Fraud Distribution')
plt.show()
