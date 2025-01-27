import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import streamlit as st

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

# Streamlit interface to allow user to change test size
st.title('Titanic Survival Prediction')
test_size = st.slider("Select Test Size", 0.1, 0.9, 0.2)  # User can change the test size here

# Split into train and test sets based on the selected test size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)  # Train the model on the new train data

# Make predictions using the newly trained model
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

# Optionally, you can plot the feature importance for better understanding
st.write("Feature Importance:")
fig, ax = plt.subplots(figsize=(10, 8))
sns.barplot(x=model.feature_importances_, y=X.columns, ax=ax)
st.pyplot(fig)

# Enhanced Age Distribution Visualization
def plot_age_distribution(df):
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df["Age"], kde=True, bins=20, color="#6c9aed", ax=ax)  # Improved color

    # Calculate statistics
    mean_age = df["Age"].mean()
    median_age = df["Age"].median()
    min_age = df["Age"].min()

    # Add vertical lines for mean, median, and minimum age
    ax.axvline(mean_age, color="green", linestyle="--", linewidth=2, label=f"Mean: {mean_age:.2f}")
    ax.axvline(median_age, color="orange", linestyle="--", linewidth=2, label=f"Median: {median_age:.2f}")
    ax.axvline(min_age, color="red", linestyle="--", linewidth=2, label=f"Min: {min_age:.2f}")

    # Customize the plot
    ax.set_title("Age Distribution of Titanic Passengers", fontsize=14)
    ax.set_xlabel("Age", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.legend(title="Statistics")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Render the plot using Streamlit
    st.pyplot(fig)

# Call the function to display the age distribution plot in Streamlit
plot_age_distribution(df)

# Fill missing 'Embarked' values with the most frequent value ("S")
df["Embarked"] = df["Embarked"].fillna("S")

# Define a color palette to use consistently across all plots
palette = {'S': '#1f77b4', 'C': '#ff7f0e', 'Q': '#2ca02c'}

# Create the figure and axes for 3 subplots with a smaller size
fig, (axis1, axis2, axis3) = plt.subplots(1, 3, figsize=(12, 4))

# Plot 1: Count plot for 'Embarked' with a consistent color palette
sns.countplot(x='Embarked', data=df, ax=axis1, palette=palette)
axis1.set_title('Count of Passengers by Embarked', fontsize=12)
axis1.set_xlabel('Embarked', fontsize=10)
axis1.set_ylabel('Count', fontsize=10)

# Plot 2: Count plot for 'Survived' with hue based on 'Embarked' and consistent colors
sns.countplot(x='Survived', hue="Embarked", data=df, order=[1, 0], ax=axis2, palette=palette)
axis2.set_title('Survival Count by Embarked', fontsize=12)
axis2.set_xlabel('Survived', fontsize=10)
axis2.set_ylabel('Count', fontsize=10)

# Plot 3: Bar plot showing survival rate by 'Embarked' with consistent colors
embark_perc = df[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).mean()
sns.barplot(x='Embarked', y='Survived', data=embark_perc, order=['S', 'C', 'Q'], ax=axis3, palette=palette)
axis3.set_title('Survival Rate by Embarked', fontsize=12)
axis3.set_xlabel('Embarked', fontsize=10)
axis3.set_ylabel('Survival Rate', fontsize=10)

# Adjust layout for better spacing
plt.tight_layout()

# Show the plots in Streamlit
st.pyplot(fig)
