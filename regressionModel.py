import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pd.read_csv('dataFrame.csv')
X = data[['CPI', 'Pandemics Cases', 'Hotel Occupancy Rate', 'GDP']]
y = data['Visitor Arrival Number']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of the Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Evaluate the model on the testing data
score = model.score(X_test, y_test)


# Print the model's coefficient and intercept
print("\n","Coefficient:", model.coef_)
print("Intercept:", model.intercept_)
print("R^2 Score:", score)
# Predict the target variable using the trained model
y_pred = model.predict(X_test)

# Plot the regression line
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual Visitor Arrival Number')
plt.ylabel('Predicted Visitor Arrival Number')
plt.title('Regression Model: Actual vs. Predicted')
plt.show()
