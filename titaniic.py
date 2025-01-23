pip install seaborn


import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# Load dataset and preprocess
titanic = sns.load_dataset('titanic')

# Fill missing values
titanic['age'].fillna(titanic['age'].median(), inplace=True)
titanic['embark_town'].fillna('Unknown', inplace=True)

# Select features and target
X = titanic[['sex', 'age', 'class', 'fare', 'embark_town']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical columns to dummy variables
y = titanic['survived']

# Train the model
model = RandomForestClassifier()
model.fit(X, y)

# Streamlit app title
st.title("Titanic Survival Prediction App")
st.write("This app predicts the survival of Titanic passengers based on their details.")

# Sidebar for user input
st.sidebar.header("Input Passenger Details")

# User input fields
sex = st.sidebar.selectbox("Sex", ['male', 'female'])
age = st.sidebar.slider("Age", 0, 100, 30)
pclass = st.sidebar.selectbox("Passenger Class", ['First', 'Second', 'Third'])
fare = st.sidebar.slider("Fare", 0.0, 600.0, 50.0)
embark_town = st.sidebar.selectbox("Embark Town", ['Southampton', 'Cherbourg', 'Queenstown', 'Unknown'])

# Encode user inputs into features
input_data = pd.DataFrame({
    'sex_male': [1 if sex == 'male' else 0],
    'age': [age],
    'class_Second': [1 if pclass == 'Second' else 0],
    'class_Third': [1 if pclass == 'Third' else 0],
    'fare': [fare],
    'embark_town_Queenstown': [1 if embark_town == 'Queenstown' else 0],
    'embark_town_Southampton': [1 if embark_town == 'Southampton' else 0],
})

# Make prediction
prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1]

# Display results
st.subheader("Prediction Result")
if prediction == 1:
    st.write("The passenger **would survive** the Titanic disaster. 🛟")
else:
    st.write("The passenger **would not survive** the Titanic disaster. 💔")

st.subheader("Survival Probability")
st.write(f"Probability of survival: **{probability * 100:.2f}%**")
