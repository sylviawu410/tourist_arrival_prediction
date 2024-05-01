import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('dataFrame.csv')


correlation_matrix = data.corr()
visitor_arrival_corr = correlation_matrix['Visitor Arrival Number']
print(visitor_arrival_corr)

start_date = pd.to_datetime('2002-01-01')  # Start date of your dataset
end_date = pd.to_datetime('2023-12-31')  # End date of your dataset

date_range = pd.date_range(start=start_date, end=end_date, freq='MS')  # Monthly frequency

# Create a new 'Date' column with the synthetic dates
data['Date'] = date_range[:len(data)]

# Extract time-based features from the 'Date' column
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Season'] = (data['Month'] - 1) // 3 + 1

# Update X with the additional time-based features
X = data[['CPI', 'Pandemics Cases', 'Hotel Occupancy Rate', 'GDP', 'Year', 'Season']]
y = data['Visitor Arrival Number']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)


# Print the model's coefficient and intercept
print("\n","Coefficient:", model.coef_)
print("Intercept:", model.intercept_)
print("R^2 Score:", score)

# Predict the target variable using the trained model
y_pred = model.predict(X_test)

def plotRegressionLine():
    plt.scatter(y_test, y_pred)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.xlabel('Actual Visitor Arrival Number')
    plt.ylabel('Predicted Visitor Arrival Number')
    plt.title('Regression Model: Actual vs. Predicted')
    plt.show()
