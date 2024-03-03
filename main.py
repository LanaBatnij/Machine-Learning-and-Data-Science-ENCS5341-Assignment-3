# Lana Batnij 1200308
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv(r"C:\Users\LENOVO\Downloads\DiabetesDataSet\diabetes_data_upload.csv")

# make sure it is all read
print("First few lines of the dataset:")
print(df.head())
print("\nLast few lines of the dataset:")
print(df.tail())

# Handle missing values
imputer = SimpleImputer(strategy='most_frequent')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# class is the target variable
X = df_imputed.drop('class', axis=1)
y = df_imputed['class']

# Visualize Age Distribution
plt.figure(figsize=(10, 6))
plt.hist(df_imputed['Age'], bins=20, color='lightpink', edgecolor='black')
plt.title('Age Distribution in the Diabetes Dataset')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# using one-hot encoding
X = pd.get_dummies(X)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets 80 training and 20 testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# k-nearest neighbors (KNN) with k=1
knn1 = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
knn1.fit(X_train, y_train)
predictions_knn1 = knn1.predict(X_test)

# k-nearest neighbors (KNN) with k=3
knn3 = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn3.fit(X_train, y_train)
predictions_knn3 = knn3.predict(X_test)

print('\nClassification Report (k=1):\n', classification_report(y_test, predictions_knn1))
print('\nClassification Report (k=3):\n', classification_report(y_test, predictions_knn3))

# Create an SVM model
svm_model = make_pipeline(StandardScaler(), SVC(C=1, gamma='auto'))

# Train the model
svm_model.fit(X_train, y_train)
# Make predictions
predictions_svm = svm_model.predict(X_test)

# Evaluate performance of the model
precision_svm = precision_score(y_test, predictions_svm)
recall_svm = recall_score(y_test, predictions_svm)
f1_svm = f1_score(y_test, predictions_svm)
accuracy_svm = accuracy_score(y_test, predictions_svm)
print(f'Accuracy (SVM): {accuracy_svm:.2f}')
print(f'Precision (SVM): {precision_svm:.2f}')
print(f'Recall (SVM): {recall_svm:.2f}')
print(f'F1-Score (SVM): {f1_svm:.2f}')

# Create a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100)

# Train and Make predictions
rf_model.fit(X_train, y_train)
predictions_rf = rf_model.predict(X_test)

# Evaluate the performance
accuracy_rf = accuracy_score(y_test, predictions_rf)
print(f'Accuracy (Random Forest): {accuracy_rf:.2f}')
predictions_rf = rf_model.predict(X_test)


# Make a confusion matrix
conf_matrix_rf = confusion_matrix(y_test, predictions_rf)
# Visualize the confusion matrix (optional)
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Set3', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Random Forest)')
plt.show()

precision = precision_score(y_test, predictions_rf)
recall = recall_score(y_test, predictions_rf)
f1 = f1_score(y_test, predictions_rf)

print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-Score: {f1:.2f}')

auc_roc = roc_auc_score(y_test, predictions_rf)
print(f'AUC-ROC: {auc_roc:.2f}')