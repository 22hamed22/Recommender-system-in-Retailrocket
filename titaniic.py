import subprocess
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Function to install required dependencies
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install necessary packages
try:
    import pandas as pd
except ImportError:
    install('pandas')

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    install('matplotlib')
    install('seaborn')

# Function to load and process data
def load_data():
    # Load Titanic dataset
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

    return df

# Enhanced Age Distribution Visualization
def plot_age_distribution(df):
    print("\nAge Distribution of Passengers:")
    
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
    
    # Show the plot
    plt.show()

# Test the function
df = load_data()
plot_age_distribution(df)
