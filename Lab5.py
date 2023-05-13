import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('lab5.csv')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1), data['target'], test_size=0.2, random_state=42)

# Generate new features
X_train['feature1'] = X_train['feature2'] + X_train['feature3']
X_test['feature1'] = X_test['feature2'] + X_test['feature3']

# Select the best features using ANOVA F-value
kbest = SelectKBest(score_func=f_classif, k=3)
X_train = kbest.fit_transform(X_train, y_train)
X_test = kbest.transform(X_test)

# Train a logistic regression model on the selected features
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Predict the target variable on the test set
y_pred = lr.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


