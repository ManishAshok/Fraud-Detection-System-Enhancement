
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv('transaction_data.csv')

# Handle missing values
data.fillna(method='ffill', inplace=True)

# Normalize numerical features
scaler = StandardScaler()
data[['transaction_amount']] = scaler.fit_transform(data[['transaction_amount']])

# Encode categorical variables
encoder = OneHotEncoder()
encoded_location = encoder.fit_transform(data[['transaction_location']])
encoded_location_df = pd.DataFrame(encoded_location.toarray(), columns=encoder.get_feature_names_out())
data = data.join(encoded_location_df)
data.drop(columns=['transaction_location'], inplace=True)

# Split data into training and testing sets
X = data.drop(['is_fraud', 'transaction_id', 'customer_id', 'transaction_date'], axis=1)
y = data['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Save the preprocessed data
X_train.to_csv('X_train.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
