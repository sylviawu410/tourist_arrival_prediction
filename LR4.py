import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
import matplotlib.pyplot as plt
data = pd.read_csv('dataFrame.csv')

start_date = pd.to_datetime('2002-01-01') 
end_date = pd.to_datetime('2023-12-31')  

date_range = pd.date_range(start=start_date, end=end_date, freq='MS')  

data['Date'] = date_range[:len(data)]

data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Quarter'] = (data['Month'] - 1) // 3 + 1


X = data[['CPI', 'Hotel Occupancy Rate', 'GDP', 'BRI','CX stock price', 'Year', 'Quarter']]
y = data['Visitor Arrival Number']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# polynimial features
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

# feature scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)

# LASSO
lasso_model = Lasso(alpha=10, max_iter=100000)  
lasso_model.fit(X_train_scaled, y_train)
lasso_score = lasso_model.score(X_test_scaled, y_test)

# details of LASSO
print("Lasso Regression Coefficients:", lasso_model.coef_)
print("Lasso Regression Intercept:", lasso_model.intercept_)
print("Lasso Regression R^2 Score:", lasso_score)


def plotPolynomialGraph():
    y_pred = lasso_model.predict(X_test_scaled)
    plt.scatter(y_test, y_pred)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.xlabel('Actual Number of Visitor Arrivals')
    plt.ylabel('Predicted Number of Visitor Arrivals')
    plt.title('Actual vs. Predicted Visitor Arrivals')
    plt.show()

# plotPolynomialGraph()