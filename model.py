# model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# 1. Load the dataset
df = pd.read_csv('Training.csv')

# 2. Preprocess the data
# The last column 'prognosis' is our target variable
X = df.drop('prognosis', axis=1)
y = df['prognosis']

# Encode the target variable (disease names) into numbers
le = LabelEncoder()
y = le.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train the model
# We'll use a Decision Tree, which is simple and effective for this task
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. Save the trained model and the label encoder
# We save them to use in our web app later
pickle.dump(model, open('disease_model.pkl', 'wb'))
pickle.dump(le, open('label_encoder.pkl', 'wb'))

print("Model training complete and files saved!")
