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
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import LabelEncoder
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
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    df = pd.read_csv(url)
    df.fillna(method='ffill', inplace=True)
    df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    df['Is_Master'] = df['Title'] == 'Master'
    df['Family_Size'] = df['SibSp'] + df['Parch']
    df['Is_Female_Dying'] = (df['Sex'] == 'female') & (df['Family_Size'] == df.groupby('Family_Size')['Survived'].transform('sum'))
    df.loc[df['Is_Master'], 'Predicted_Survival'] = 1
    df.loc[df['Is_Female_Dying'], 'Predicted_Survival'] = 0
    df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    labelencoder = LabelEncoder()
    df['Sex'] = labelencoder.fit_transform(df['Sex'])
    df['Embarked'] = labelencoder.fit_transform(df['Embarked'])
    df['Title'] = labelencoder.fit_transform(df['Title'])

    X = df.drop(columns=['Survived', 'Predicted_Survival'])
    y = df['Survived']
    
    return df, X, y

# Define function to plot age and survival
def plot_age_survival(df):
    train_data = df.iloc[:891]
    train_data['Survived'] = train_data['Survived'].replace({0: 'Not Survived', 1: 'Survived'})

    plt.figure(figsize=(12, 6))
    sns.histplot(data=train_data, x='Age', hue='Survived', multiple='stack',
                 palette={'Not Survived': '#ff9999', 'Survived': '#66b3ff'}, bins=20)

    g = sns.FacetGrid(train_data, col="Sex", height=5, aspect=1.2, palette="Set2")
    g.map_dataframe(sns.histplot, x="Age", hue="Survived", multiple="stack",
                    palette={'Not Survived': '#ff9999', 'Survived': '#66b3ff'}, bins=20)
    g.set_axis_labels("Age", "Count")
    g.set_titles("{col_name} Passengers")
    g.add_legend(title="Survival Status")
    g.tight_layout()

    plt.show()

# Define function to plot embarkment fare
def plot_embarkment_fare(df):
    embark_fare = df[(df['PassengerId'] != 62) & (df['PassengerId'] != 830)]
    embark_fare['Pclass'] = embark_fare['Pclass'].replace({1: 'Class 1', 2: 'Class 2', 3: 'Class 3'})
    embark_fare['Embarked'] = embark_fare['Embarked'].replace({'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'})

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=embark_fare, x='Embarked', y='Fare', hue='Pclass', palette='pastel')
    plt.axhline(y=80, color='red', linestyle='dashed', linewidth=2, label='y = £80')
    plt.title("Fare by Embarkment Port and Passenger Class")
    plt.xlabel("Embarkment Port")
    plt.ylabel("Fare (£)")
    plt.legend(title="Passenger Class")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Define function to plot fare density
def plot_fare_density(df):
    filtered_df = df[(df['Pclass'] == 3) & (df['Embarked'] == 'S')]
    median_fare = filtered_df['Fare'].median()

    plt.figure(figsize=(8, 6))
    sns.kdeplot(filtered_df['Fare'], fill=True, color='#ffcc99', alpha=0.4, label='Density')
    plt.axvline(median_fare, color='blue', linestyle='dashed', linewidth=1.5, label=f'Median Fare: £{median_fare:.2f}')
    plt.title("Fare Density for Pclass 3 Passengers Embarked at S")
    plt.xlabel("Fare (£)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()

# Define function to plot embarkment related visualizations
def plot_embarkment_related(df):
    df["Embarked"] = df["Embarked"].fillna("S")
    palette = {'S': '#1f77b4', 'C': '#ff7f0e', 'Q': '#2ca02c'}

    fig, (axis1, axis2, axis3) = plt.subplots(1, 3, figsize=(12, 4))

    sns.countplot(x='Embarked', data=df, ax=axis1, palette=palette)
    axis1.set_title('Count of Passengers by Embarked', fontsize=12)
    axis1.set_xlabel('Embarked', fontsize=10)
    axis1.set_ylabel('Count', fontsize=10)

    sns.countplot(x='Survived', hue="Embarked", data=df, order=[1, 0], ax=axis2, palette=palette)
    axis2.set_title('Survival Count by Embarked', fontsize=12)
    axis2.set_xlabel('Survived', fontsize=10)
    axis2.set_ylabel('Count', fontsize=10)

    embark_perc = df[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).mean()
    sns.barplot(x='Embarked', y='Survived', data=embark_perc, order=['S', 'C', 'Q'], ax=axis3, palette=palette)
    axis3.set_title('Survival Rate by Embarked', fontsize=12)
    axis3.set_xlabel('Embarked', fontsize=10)
    axis3.set_ylabel('Survival Rate', fontsize=10)

    plt.tight_layout()
    plt.show()

# Enhanced Age Distribution Visualization
def plot_age_distribution(df):
    print("\nAge Distribution of Passengers:")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df["Age"], kde=True, bins=20, color="#6c9aed", ax=ax)
    
    mean_age = df["Age"].mean()
    median_age = df["Age"].median()
    min_age = df["Age"].min()
    
    ax.axvline(mean_age, color="green", linestyle="--", linewidth=2, label=f"Mean: {mean_age:.2f}")
    ax.axvline(median_age, color="orange", linestyle="--", linewidth=2, label=f"Median: {median_age:.2f}")
    ax.axvline(min_age, color="red", linestyle="--", linewidth=2, label=f"Min: {min_age:.2f}")
    
    ax.set_title("Age Distribution of Titanic Passengers", fontsize=14)
    ax.set_xlabel("Age", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.legend(title="Statistics")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    
    plt.show()

# Streamlit app layout
st.title("Titanic Survival Prediction App")

# Load data
df, X, y = load_data()

# Allow user to modify parameters for model
st.sidebar.title("Model Parameters")
n_estimators = st.sidebar.slider("Number of Estimators (n_estimators)", 10, 200, 100)
test_size = st.sidebar.slider("Test Size", 0.1, 0.9, 0.2)

# Split into train and test sets based on the selected test size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Button to train and evaluate model
if st.button("Train Model"):
    accuracy = train_and_evaluate(X_train, X_test, y_train, y_test, n_estimators)
    st.write(f"Model Accuracy: {accuracy*100:.2f}%")

# Display the dataset
st.write("Titanic Dataset Preview:")
st.dataframe(X.head())

# Cross-validation result
st.subheader("Cross-Validation (5-fold) Accuracy:")
cv_accuracy = cross_val(X, y, n_estimators)
st.write(f"Average Cross-Validation Accuracy: {cv_accuracy * 100:.2f}%")

# Plot various visualizations
st.subheader("Age and Survival Distribution")
plot_age_survival(df)

st.subheader("Embarkment and Fare Distribution")
plot_embarkment_fare(df)

st.subheader("Fare Density for Pclass 3 Passengers Embarked at S")
plot_fare_density(df)

st.subheader("Embarkment Related Visualizations")
plot_embarkment_related(df)

st.subheader("Age Distribution of Passengers")
plot_age_distribution(df)
