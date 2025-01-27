# titanic_app.py

import subprocess
import sys

# Function to install required dependencies
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install necessary packages
try:
    import pandas as pd
except ImportError:
    install('pandas')

try:
    import streamlit as st
except ImportError:
    install('streamlit')

try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
except ImportError:
    install('scikit-learn')

# Install visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    install('matplotlib')
    install('seaborn')

# Define function to load and process data
def load_data():
    # Load Titanic dataset (ensure the dataset is in the working directory or use a URL)
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    df = pd.read_csv(url)
    
    # Basic data processing: handling missing values
    df.fillna(method='ffill', inplace=True)
    
    # Select features and target
    X = df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex']]
    X = pd.get_dummies(X, drop_first=True)
    y = df['Survived']
    
    return df, X, y

# Define function for training and evaluating the model
def train_and_evaluate(X_train, X_test, y_train, y_test, n_estimators):
    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

# Streamlit app layout
st.title("Titanic Survival Prediction App")

# Load data
df, X, y = load_data()

# Allow user to modify parameters for model
st.sidebar.title("Model Parameters")
n_estimators = st.sidebar.slider("Number of Estimators (n_estimators)", 10, 200, 100)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Button to train and evaluate model
if st.button("Train Model"):
    accuracy = train_and_evaluate(X_train, X_test, y_train, y_test, n_estimators)
    st.write(f"Model Accuracy: {accuracy*100:.2f}%")

# Display the dataset
st.write("Titanic Dataset Preview:")
st.dataframe(X.head())

# Plot Age Distribution
st.subheader("Age Distribution of Passengers")
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(df['Age'], kde=True, bins=20, ax=ax)
ax.set_title('Age Distribution of Titanic Passengers')
ax.set_xlabel('Age')
ax.set_ylabel('Frequency')
st.pyplot(fig)

# Plot Pclass Distribution (which is related to Ticket class)
st.subheader("Distribution of Passengers by Pclass")
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(data=df, x='Pclass', ax=ax)
ax.set_title('Number of Passengers by Pclass (Ticket Class)')
ax.set_xlabel('Pclass')
ax.set_ylabel('Count')
st.pyplot(fig)
