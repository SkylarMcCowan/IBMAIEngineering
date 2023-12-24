import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import requests

def download(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)

# Download the dataset
path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillUp/labs/ML-FinalAssignment/Weather_Data.csv'
download(path, "Weather_Data.csv")
            
# Load and process the dataset
filename = "Weather_Data.csv"
df = pd.read_csv(filename)
df_sydney_processed = pd.get_dummies(data=df, columns=['RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])
df_sydney_processed.replace(['No', 'Yes'], [0, 1], inplace=True)
df_sydney_processed.drop('Date', axis=1, inplace=True)
df_sydney_processed = df_sydney_processed.astype(float)
features = df_sydney_processed.drop(columns='RainTomorrow', axis=1)
Y = df_sydney_processed['RainTomorrow']
            
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, Y, test_size=0.2, random_state=10)

# Create and train a Linear Regression model
LinearReg = LinearRegression()
LinearReg.fit(X_train, y_train)

# Use the predict method on the testing data
predictions = LinearReg.predict(X_test)

# Optionally, print the first few predictions
print("First few predictions:", predictions[:5])

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error (MAE):", mae)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error (MSE):", mse)

# Calculate R-squared Score
r2 = r2_score(y_test, predictions)
print("R-squared Score:", r2)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error (MAE):", mae)

mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error (MSE):", mse)

r2 = r2_score(y_test, predictions)
print("R-squared Score:", r2)
