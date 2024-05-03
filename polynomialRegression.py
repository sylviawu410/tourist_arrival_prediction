import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('dataFrame.csv')

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

# polynimial regression
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)
model = LinearRegression()
model.fit(X_train_poly, y_train)
score = model.score(X_test_poly, y_test)

# Print the model's coefficient and intercept
print("\nCoefficient:", model.coef_)
print("Intercept:", model.intercept_)
print("R^2 Score:", score)

def plotPolynomialGraph():
    # Generate a range of values for the x-axis by selecting the 'CPI' column from the original data
    x_range = np.linspace(X['CPI'].min(), X['CPI'].max(), 100)

    # Create a DataFrame with the x-axis values for all features
    x_range_df = pd.DataFrame(columns=X.columns, data=np.zeros((100, len(X.columns))))
    x_range_df['CPI'] = x_range

    # Transform the x-axis DataFrame into polynomial features
    x_range_poly = poly_features.transform(x_range_df)

    # Make predictions using the trained model and the polynomial features
    y_range_pred = model.predict(x_range_poly)

    # Plot the original data points
    plt.scatter(X['CPI'], y, color='blue', label='Original Data')

    # Plot the polynomial regression line
    plt.plot(x_range, y_range_pred, color='red', label='Polynomial Regression')

    plt.xlabel('CPI')
    plt.ylabel('Visitor Arrival Number')
    plt.legend()

    plt.show()