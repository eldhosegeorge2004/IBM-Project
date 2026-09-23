# Project: AutoAnalytics App

## Overview
The AutoAnalytics App is a comprehensive Streamlit-based web application that automates the Exploratory Data Analysis (EDA) process and offers a robust machine learning module for predictive modeling. Built as part of the IBM SkillsBuild Data Analytics with AI Academic Internship Program.

## Dataset
This app supports arbitrary generic CSV and Excel datasets from Kaggle.
Example dataset used for testing: [Titanic Dataset](https://www.kaggle.com/c/titanic/data) or any retail sales dataset.

## Technologies Used
- **Python 3**
- **Streamlit**: For the interactive web interface.
- **Pandas**: For robust data manipulation and cleaning.
- **Plotly Express**: For interactive visualization and charting.
- **Scikit-learn**: For predictive modeling (Logistic Regression).

## Setup & Run Instructions
1. Install Python 3.8+ if not already installed.
2. Clone or download this project folder.
3. Install the dependencies by running:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   streamlit run EldhoseGeorge_AutoAnalytics.py
   ```
5. Navigate to the local URL (typically http://localhost:8501) provided in your terminal.

## Key Information
- **Upload & Profile**: Supports CSV (utf-8 and latin-1 encodings) and Excel formats.
- **Cleaning**: Out-of-the-box imputation of missing values (mean/mode), duplicate handling, and action logging.
- **Compare & Chart**: Dynamic UI for Scatter Plots, Bar Charts, and Line Charts.
- **Insights**: Group-by summations for categorical metrics.
- **Predictive Model**: Train a Logistic Regression model securely in the browser.
