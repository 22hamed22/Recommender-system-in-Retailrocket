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
    from sklearn.preprocessing import LabelEncoder  # Add this import for LabelEncoder
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
    
    # Feature engineering
    df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    df['Is_Master'] = df['Title'] == 'Master'
    df['Family_Size'] = df['SibSp'] + df['Parch']
    df['Is_Female_Dying'] = (df['Sex'] == 'female') & (df['Family_Size'] == df.groupby('Family_Size')['Survived'].transform('sum'))
    df.loc[df['Is_Master'], 'Predicted_Survival'] = 1
    df.loc[df['Is_Female_Dying'], 'Predicted_Survival'] = 0

    # Drop unnecessary columns
    df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    # Encode categorical features
    labelencoder = LabelEncoder()
    df['Sex'] = labelencoder.fit_transform(df['Sex'])  # 'Male' -> 1, 'Female' -> 0
    df['Embarked'] = labelencoder.fit_transform(df['Embarked'])  # Convert 'S', 'C', 'Q' to numeric
    df['Title'] = labelencoder.fit_transform(df['Title'])  # Convert 'Mr', 'Mrs', 'Miss' to numeric

    # Define features (X) and target (y)
    X = df.drop(columns=['Survived', 'Predicted_Survival'])
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
test_size = st.sidebar.slider("Test Size", 0.1, 0.9, 0.2)  # User can change the test size here

# Split into train and test sets based on the selected test size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

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
