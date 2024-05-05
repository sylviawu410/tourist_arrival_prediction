import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
import matplotlib.pyplot as plt

data = pd.read_csv('dataFrame.csv')

start_date = pd.to_datetime('2002-01-01')  # Start date of the datasets
end_date = pd.to_datetime('2023-12-31')  
date_range = pd.date_range(start=start_date, end=end_date, freq='MS')  

# Create a new 'Date' column with the synthetic dates
data['Date'] = date_range[:len(data)]

# Extract time-based features from the 'Date' column
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Quarter'] = (data['Month'] - 1) // 3 + 1

X = data[['CPI', 'Pandemics Cases', 'Hotel Occupancy Rate', 'GDP', 'BRI','CX stock price', 'Year', 'Quarter']]
y = data['Visitor Arrival Number']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Linear Regression model
model = LinearRegression(fit_intercept= True)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)

# Lasso Regression model
lasso_model = Lasso(alpha=0.001)  
lasso_model.fit(X_train, y_train)
lasso_score = lasso_model.score(X_test, y_test)

# Print the model's coefficient and intercept
print("Coefficient:", model.coef_)
print("Intercept:", model.intercept_)
print("R^2 Score:", score)

print()
print("Lasso Regression Coefficients:", lasso_model.coef_)
print("Lasso Regression Intercept:", lasso_model.intercept_)
print("Lasso Regression R^2 Score:", lasso_score)

# Predict the target variable using the trained model
y_pred = model.predict(X_test)

def plotRegressionLine():
    plt.scatter(y_test, y_pred)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.xlabel('Actual Visitor Arrival Number')
    plt.ylabel('Predicted Visitor Arrival Number')
    plt.title('Regression Model: Actual vs. Predicted')
    plt.show()
# plotRegressionLine()