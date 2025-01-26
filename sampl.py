import subprocess
import sys

# Function to install a package
def install_package(package):
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    else:
        print(f"{package} is already installed.")

# List of required packages
required_packages = ["streamlit", "pandas", "numpy", "matplotlib"]

# Install each package
for package in required_packages:
    install_package(package)

# Your Streamlit app starts here
import streamlit as st
import pandas as pd
import numpy as np

st.title("Streamlit App with Embedded Dependencies")
st.write("All required packages have been installed!")
