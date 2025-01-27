import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
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
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

# Streamlit layout
st.title("Titanic Survival Prediction App")

# Display the dataset
st.write("Titanic Dataset Preview:")
st.dataframe(df.head())

# Feature importance visualization
st.subheader("Feature Importance")
features = X.columns
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax)
ax.set_title("Random Forest Feature Importance")
st.pyplot(fig)

# Age Distribution Plot
st.subheader("Age Distribution of Titanic Passengers")
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(df['Age'], kde=True, bins=20, ax=ax)
ax.set_title('Age Distribution of Titanic Passengers')
ax.set_xlabel('Age')
ax.set_ylabel('Frequency')
st.pyplot(fig)

# Embarked and Survival Rate Plot
st.subheader("Survival Rates by Embarkment Port")
sns.countplot(x='Embarked', data=df, hue='Survived')
plt.title('Survival Rate by Embarkment Port')
plt.xlabel('Embarked')
plt.ylabel('Survival Count')
st.pyplot(plt)

# Survival Prediction for Specific Passenger Example (Optional)
st.subheader("Predict Survival for a Specific Passenger")

# Sample data input
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
age = st.slider("Age", 0, 100, 30)
sibsp = st.slider("Siblings/Spouses Aboard (SibSp)", 0, 10, 0)
parch = st.slider("Parents/Children Aboard (Parch)", 0, 10, 0)
sex = st.selectbox("Sex", ['male', 'female'])

# Encode input for prediction
sex_encoded = labelencoder.transform([sex])[0]
input_data = np.array([[pclass, age, sibsp, parch, sex_encoded]])
input_data = pd.DataFrame(input_data, columns=X.columns)

# Prediction
if st.button("Predict Survival"):
    survival_pred = model.predict(input_data)
    survival = "Survived" if survival_pred == 1 else "Did Not Survive"
    st.write(f"The prediction for this passenger is: {survival}")
