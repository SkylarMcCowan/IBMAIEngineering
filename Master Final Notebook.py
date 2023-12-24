import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score, log_loss
import requests

# Function to download the dataset
def download(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)

# Downloading the dataset
print("Downloading the dataset...")
path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillUp/labs/ML-FinalAssignment/Weather_Data.csv'
download(path, "Weather_Data.csv")

# Loading and preprocessing the dataset
print("Processing the dataset...")
df = pd.read_csv("Weather_Data.csv")
df_processed = pd.get_dummies(data=df, columns=['RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])
df_processed.replace(['No', 'Yes'], [0, 1], inplace=True)
df_processed.drop('Date', axis=1, inplace=True)
df_processed = df_processed.astype(float)
features = df_processed.drop(columns='RainTomorrow', axis=1)
Y = df_processed['RainTomorrow']

# Splitting the dataset
print("Splitting the dataset...")
X_train, X_test, y_train, y_test = train_test_split(features, Y, test_size=0.2, random_state=1)

# Training multiple models
print("Training models...")

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
predictions_lr = lr.predict(X_test)

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, y_train)
predictions_knn = knn.predict(X_test)

# Decision Tree
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
predictions_tree = tree.predict(X_test)

# Logistic Regression
logreg = LogisticRegression(solver='liblinear')
logreg.fit(X_train, y_train)
predictions_logreg = logreg.predict(X_test)
predict_proba_logreg = logreg.predict_proba(X_test)

# Support Vector Machine
svm = SVC()
svm.fit(X_train, y_train)
predictions_svm = svm.predict(X_test)

# Evaluating model performance
print("Evaluating models...")

metrics_data = []

# Models and their predictions (excluding Linear Regression for classification metrics)
models = ['KNN', 'Decision Tree', 'Logistic Regression', 'SVM']
model_predictions = [predictions_knn, predictions_tree, predictions_logreg, predictions_svm]

for i, model in enumerate(models):
    accuracy = accuracy_score(y_test, model_predictions[i])
    jaccard = jaccard_score(y_test, model_predictions[i], average='macro')
    f1 = f1_score(y_test, model_predictions[i], average='macro')
    logloss = 'N/A'
    if model == 'Logistic Regression':
        logloss = log_loss(y_test, predict_proba_logreg)
    
    metrics_data.append({'Model': model, 'Accuracy': accuracy, 'Jaccard': jaccard, 'F1-Score': f1, 'LogLoss': logloss})

# Convert list of dictionaries to DataFrame
metrics = pd.DataFrame(metrics_data)

# Displaying the evaluation metrics
print("\nModel Evaluation Metrics:")
print(metrics)