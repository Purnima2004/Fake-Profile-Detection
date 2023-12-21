# Called necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC

# Loaded the dataset
df = pd.read_csv('fake profiles.csv')

# Selected specific columns based on conditions
x = df[['profile pic', 'nums/length username', 'fullname words', 'nums/length fullname', 'name==username','external URL','private']]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x[['profile pic', 'nums/length username', 'fullname words', 'nums/length fullname', 'name==username','external URL','private']]= scaler.fit_transform(x[['profile pic', 'nums/length username', 'fullname words', 'nums/length fullname', 'name==username','external URL','private']])

# Selected the target variable 'fake'
y = df[['fake']]

# Apply SelectKBest class to extract 10 features
bestfeatures = SelectKBest(score_func=chi2, k=7)
x = bestfeatures.fit_transform(x, y)  # Transform x using the selected features
# Now, x only contains the selected features.

#Splitting the datasets into training and testing
x_train , x_test , y_train , y_test= train_test_split(x, y, test_size= 0.4, random_state=45)

# Initialized and trained the SVM model
model = SVC(kernel='sigmoid', degree=2, C=3, gamma='auto', random_state=0)
model.fit(x_train, y_train)

# Prediction on the test set
y_pred = model.predict(x_test)

# Calculating accuracy and printed classification report
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

report = classification_report(y_test, y_pred)
print(f'Classification Report:\n{report}')