import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the Titanic dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

# Feature engineering (based on the rules)
df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
df['Is_Master'] = df['Title'] == 'Master'

# Rule for females whose entire family, excluding adult males, all die
df['Family_Size'] = df['SibSp'] + df['Parch']
df['Is_Female_Dying'] = (df['Sex'] == 'female') & (df['Family_Size'] == df.groupby('Family_Size')['Survived'].transform('sum'))

# Apply Rules (Override predictions where rules apply)
df.loc[df['Is_Master'], 'Predicted_Survival'] = 1
df.loc[df['Is_Female_Dying'], 'Predicted_Survival'] = 0

# Fill missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop unnecessary columns
df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Encode categorical features
labelencoder = LabelEncoder()
df['Sex'] = labelencoder.fit_transform(df['Sex'])  # 'Male' -> 1, 'Female' -> 0
df['Embarked'] = labelencoder.fit_transform(df['Embarked'])  # Convert 'S', 'C', 'Q' to numeric

# Optionally, you can encode Title column with LabelEncoder as well
df['Title'] = labelencoder.fit_transform(df['Title'])  # Convert 'Mr', 'Mrs', 'Miss' to numeric

# Define features (X) and target (y)
X = df.drop(columns=['Survived', 'Predicted_Survival'])
y = df['Survived']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model - RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
