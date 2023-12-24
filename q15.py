import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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
df = pd.read_csv("Weather_Data.csv")
df_sydney_processed = pd.get_dummies(data=df, columns=['RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])
df_sydney_processed.replace(['No', 'Yes'], [0, 1], inplace=True)
df_sydney_processed.drop('Date', axis=1, inplace=True)
df_sydney_processed = df_sydney_processed.astype(float)
features = df_sydney_processed.drop(columns='RainTomorrow', axis=1)
Y = df_sydney_processed['RainTomorrow']
            
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, Y, test_size=0.2, random_state=1)

# Create and train a Logistic Regression model
LR = LogisticRegression(solver='liblinear')
LR.fit(X_train, y_train)
print("Logistic Regression model trained with solver 'liblinear'")

# Use the predict method on the testing data
predictions = LR.predict(X_test)
predict_proba = LR.predict_proba(X_test)

# Optionally, print the first few predictions and probabilities
print("First few predictions from the Logistic Regression model:", predictions[:5])
print("First few prediction probabilities from the Logistic Regression model:", predict_proba[:5])

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss

# Calculate Accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Calculate Precision
precision = precision_score(y_test, predictions)
print("Precision:", precision)

# Calculate Recall
recall = recall_score(y_test, predictions)
print("Recall:", recall)

# Calculate F1 Score
f1 = f1_score(y_test, predictions)
print("F1 Score:", f1)

# Calculate Log Loss
logloss = log_loss(y_test, predict_proba)
print("Log Loss:", logloss)


from sklearn.svm import SVC

# Create an SVM model
SVM = SVC()

# Train the model using the training sets
SVM.fit(X_train, y_train)

# Optionally, you can print a statement to confirm the training is complete
print("SVM model trained")



# Use the predict method on the testing data
predictions = SVM.predict(X_test)

# Optionally, you can print the first few predictions to verify
print("First few predictions from the SVM model:", predictions[:5])